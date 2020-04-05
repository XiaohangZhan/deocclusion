% Update your test set path here
trimap_path = '/home/liyaoyi/dataset/Adobe/Combined_Dataset/Test_set/trimaps';
alpha_path = '/home/liyaoyi/dataset/Adobe/Combined_Dataset/Test_set/alpha_copy';
pred_path = '/home/liyaoyi/Source/python/GCA-Matting/prediction/gca-dist/';

alpha_list = dir(alpha_path);
alpha_list = alpha_list(3:end);

total_sad_loss = 0;
total_mse_loss = 0;
total_grad_loss = 0;
total_conn_loss = 0;

for i = 1:length(alpha_list)
    trimap = imread([trimap_path, '/', alpha_list(i).name]);
    target = imread([alpha_path, '/', alpha_list(i).name]);
    target = target(:,:,1);
    pred = imread([pred_path, '/', alpha_list(i).name]);

    sad = compute_sad_loss(pred,target,trimap);
    mse = compute_mse_loss(pred,target,trimap);
    grad = compute_gradient_loss(pred,target,trimap)/1000;
    conn = compute_connectivity_error(pred,target,trimap,0.1)/1000;

    total_sad_loss = total_sad_loss + sad;
    total_mse_loss = total_mse_loss + mse;
    total_grad_loss = total_grad_loss + grad;
    total_conn_loss = total_conn_loss + conn;

    disp([alpha_list(i).name, ' SAD: ', num2str(sad), ' MSE: ', num2str(mse), ' GRAD: ', num2str(grad), ' CONN: ', num2str(conn)]);
end

image_num = length(alpha_list);
disp(['MEAN: ', ' SAD: ', num2str(total_sad_loss/image_num), ' MSE: ', num2str(total_mse_loss/image_num), ' GRAD: ', num2str(total_grad_loss/image_num), ' CONN: ', num2str(total_conn_loss/image_num)]);
