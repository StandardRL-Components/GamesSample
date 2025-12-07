import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Balance a rotating platform to stack falling blocks. The higher your tower, "
        "the higher your score, but be careful not to let any blocks fall off!"
    )
    user_guide = (
        "Controls: Use the ← and → arrow keys to rotate the platform and catch the falling blocks."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Assumed FPS for physics and animation timing

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (20, 40, 80)
        self.COLOR_PLATFORM = (180, 180, 190)
        self.COLOR_PLATFORM_SIDE = (120, 120, 130)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.BLOCK_COLORS = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (100, 255, 255), (255, 100, 255)
        ]

        # Game Parameters
        self.PLATFORM_CENTER = (self.WIDTH // 2, self.HEIGHT - 50)
        self.PLATFORM_LENGTH = 200
        self.PLATFORM_THICKNESS = 15
        self.ROTATION_SPEED = 2.0  # degrees per step
        self.INITIAL_GRAVITY = 0.5
        self.GRAVITY_INCREASE_INTERVAL = 600 # 20 seconds at 30fps
        self.GRAVITY_INCREASE_AMOUNT = 0.05
        self.INITIAL_BLOCK_SIZE = 25
        self.BLOCK_SIZE_INCREASE_INTERVAL = 10 # successful landings
        self.BLOCK_SIZE_INCREASE_FACTOR = 1.05
        self.WIN_SCORE = 50
        self.MAX_STEPS = 10000

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # State variables are initialized in reset()
        self.platform_angle = 0.0
        self.gravity = 0.0
        self.current_block_size = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.landed_blocks = []
        self.falling_block = None
        self.landings_since_size_increase = 0
        self.particles = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.platform_angle = 0.0
        self.gravity = self.INITIAL_GRAVITY
        self.current_block_size = self.INITIAL_BLOCK_SIZE
        
        self.landed_blocks = []
        self.falling_block = None
        self.landings_since_size_increase = 0
        self.particles = []
        
        self._spawn_new_block()
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        # 1. Handle Input
        movement = action[0]
        if movement == 3:  # Left
            self.platform_angle -= self.ROTATION_SPEED
        elif movement == 4:  # Right
            self.platform_angle += self.ROTATION_SPEED
        self.platform_angle %= 360

        # 2. Update Game State
        self.steps += 1
        
        # Update continuous reward
        reward += 0.01 * len(self.landed_blocks)

        # Update difficulty
        if self.steps > 0 and self.steps % self.GRAVITY_INCREASE_INTERVAL == 0:
            self.gravity += self.GRAVITY_INCREASE_AMOUNT

        # Update particles
        self._update_particles()
        
        # Update falling block
        if self.falling_block:
            landing_info = self._update_falling_block()
            if landing_info:
                # Block landed
                self.landed_blocks.append(landing_info['block'])
                self.falling_block = None
                
                # Calculate landing reward
                dist_from_center = abs(landing_info['block']['relative_pos'][0])
                landing_reward = max(1.0, 3.0 - (dist_from_center / (self.PLATFORM_LENGTH / 4)))
                reward += landing_reward
                self.score += round(landing_reward)
                
                self._spawn_particles(landing_info['world_pos'], landing_info['block']['color'])
                
                # Check for block size increase
                self.landings_since_size_increase += 1
                if self.landings_since_size_increase >= self.BLOCK_SIZE_INCREASE_INTERVAL:
                    self.current_block_size *= self.BLOCK_SIZE_INCREASE_FACTOR
                    self.landings_since_size_increase = 0

                self._spawn_new_block()
            
            elif self.falling_block['pos'][1] > self.HEIGHT + self.falling_block['size']:
                # Block fell off
                self.game_over = True

        # 3. Check Termination Conditions
        terminated = self.game_over or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over: # Win or max steps
             self.game_over = True
        
        if self.game_over and not (self.score >= self.WIN_SCORE):
            reward = -10.0 # Penalty for losing
        elif self.score >= self.WIN_SCORE:
            reward += 100.0
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Truncated is always False as per current logic
            self._get_info()
        )

    def _get_observation(self):
        self._render_background()
        self._render_particles()
        self._render_platform()
        self._render_blocks()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame and numpy have different coordinate systems.
        # Pygame: (width, height), Numpy: (height, width)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _spawn_new_block(self):
        x_spawn = self.np_random.uniform(self.WIDTH / 2 - 50, self.WIDTH / 2 + 50)
        self.falling_block = {
            'pos': [x_spawn, -self.current_block_size],
            'vel_y': 0,
            'size': self.current_block_size,
            'color': random.choice(self.BLOCK_COLORS),
        }
        
    def _update_falling_block(self):
        block = self.falling_block
        
        # Store previous position for collision detection
        prev_y = block['pos'][1]
        
        # Apply gravity
        block['vel_y'] += self.gravity * (1.0 / self.FPS)
        block['pos'][1] += block['vel_y']
        
        # --- Collision Detection (using rotated coordinate system) ---
        theta = -math.radians(self.platform_angle) # Negative for rotating point
        
        # Rotate block's previous and current center points into platform's frame
        px, py = self.PLATFORM_CENTER
        prev_rotated = self._rotate_point(block['pos'][0] - px, prev_y - py, theta)
        curr_rotated = self._rotate_point(block['pos'][0] - px, block['pos'][1] - py, theta)

        platform_top_y = -self.PLATFORM_THICKNESS / 2
        block_bottom_y = -block['size'] / 2 # In its own frame

        # Check if block's bottom edge crossed the platform's top surface
        if prev_rotated[1] < platform_top_y + block_bottom_y and \
           curr_rotated[1] >= platform_top_y + block_bottom_y:
            
            # Check if it landed within the horizontal bounds of the platform
            if abs(curr_rotated[0]) <= self.PLATFORM_LENGTH / 2:
                # Successful landing
                landed_block = {
                    'relative_pos': (curr_rotated[0], platform_top_y + block_bottom_y),
                    'size': block['size'],
                    'color': block['color'],
                }
                
                # Calculate world position for particle effects
                world_pos_rotated = self._rotate_point(landed_block['relative_pos'][0], landed_block['relative_pos'][1], -theta)
                world_pos = (world_pos_rotated[0] + px, world_pos_rotated[1] + py)
                
                return {'block': landed_block, 'world_pos': world_pos}
        return None

    def _rotate_point(self, x, y, theta):
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        return x * cos_t - y * sin_t, x * sin_t + y * cos_t

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Particle gravity
            p['life'] -= 1

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_platform(self):
        self._draw_3d_rect(
            self.PLATFORM_CENTER,
            (self.PLATFORM_LENGTH, self.PLATFORM_THICKNESS),
            self.platform_angle,
            self.COLOR_PLATFORM,
            self.COLOR_PLATFORM_SIDE
        )

    def _render_blocks(self):
        # Render landed blocks
        theta = math.radians(self.platform_angle)
        for block in self.landed_blocks:
            rel_pos = block['relative_pos']
            
            # Rotate relative position back to world space orientation
            rotated_rel_pos = self._rotate_point(rel_pos[0], rel_pos[1], -theta) # use negative theta to un-rotate
            
            world_pos = (
                self.PLATFORM_CENTER[0] + rotated_rel_pos[0],
                self.PLATFORM_CENTER[1] + rotated_rel_pos[1]
            )
            self._draw_3d_rect(world_pos, (block['size'], block['size']), self.platform_angle, block['color'])

        # Render falling block
        if self.falling_block:
            block = self.falling_block
            self._draw_3d_rect(block['pos'], (block['size'], block['size']), 0, block['color'])

    def _draw_3d_rect(self, center, dims, angle, color_top, color_side=None):
        w, h = dims
        if color_side is None:
            color_side = (color_top[0] * 0.7, color_top[1] * 0.7, color_top[2] * 0.7)
        
        # Define 4 corners of the top face in local space
        points = [
            np.array([-w / 2, -h / 2]), np.array([w / 2, -h / 2]),
            np.array([w / 2, h / 2]), np.array([-w / 2, h / 2])
        ]
        
        # Rotate points
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated_points = [rot_matrix @ p for p in points]
        
        # Define depth/side offset
        side_offset = np.array([0, self.PLATFORM_THICKNESS if h > self.PLATFORM_THICKNESS else h])
        
        # Calculate screen positions
        top_face_pts = [p + center for p in rotated_points]
        bottom_face_pts = [p + center + side_offset for p in rotated_points]
        
        # Draw side faces
        for i in range(4):
            p1 = top_face_pts[i]
            p2 = top_face_pts[(i + 1) % 4]
            p3 = bottom_face_pts[(i + 1) % 4]
            p4 = bottom_face_pts[i]
            side_poly = [tuple(map(int, p)) for p in [p1, p2, p3, p4]]
            pygame.gfxdraw.filled_polygon(self.screen, side_poly, color_side)
            pygame.gfxdraw.aapolygon(self.screen, side_poly, color_side)
            
        # Draw top face
        top_poly_int = [tuple(map(int, p)) for p in top_face_pts]
        pygame.gfxdraw.filled_polygon(self.screen, top_poly_int, color_top)
        pygame.gfxdraw.aapolygon(self.screen, top_poly_int, color_top)

    def _render_particles(self):
        for p in self.particles:
            size = max(1, p['life'] / 10.0 * p['size'])
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(size))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            if self.score >= self.WIN_SCORE:
                end_text_str = "YOU WIN!"
                color = (150, 255, 150)
            else:
                end_text_str = "GAME OVER"
                color = (255, 150, 150)
            
            end_text = self.font_game_over.render(end_text_str, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block will not be run by the autograder but is useful for testing.
    # Un-comment the next line to run in a window instead of headless
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    
    # Manual play loop
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Balance Block")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    
    action = [0, 0, 0] # [movement, space, shift]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action space
        action = [0, 0, 0]
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()

        clock.tick(env.FPS)
        
    env.close()