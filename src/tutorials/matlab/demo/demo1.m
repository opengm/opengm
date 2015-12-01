function L = demo1(I, color, T)
    I = double(I);
    d = abs(I(:,:,1)-color(1)) +  abs(I(:,:,2)-color(2)) +  abs(I(:,:,3)-color(3));
    L = d<T;
end