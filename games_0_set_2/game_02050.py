
# Generated: 2025-08-27T19:06:22.647288
# Source Brief: brief_02050.md
# Brief Index: 2050

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the block. Press space to drop it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack blocks to reach towering heights in this isometric arcade game. "
        "Position the moving block and drop it carefully. The landing area shrinks as you build higher!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width = 640
        self.screen_height = 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Visual & Game Constants ---
        self.FONT = pygame.font.Font(None, 36)
        self.COLOR_BG = pygame.Color("#2c3e50")
        self.COLOR_GRID = pygame.Color("#34495e")
        self.COLOR_LANDING_ZONE = pygame.Color("#2ecc71")
        self.BLOCK_COLORS = [pygame.Color(c) for c in ['#e74c3c', '#f1c40f', '#3498db', '#9b59b6', '#2ecc71', '#e67e22', '#1abc9c', '#d35400', '#2980b9', '#c0392b']]
        
        # Game mechanics
        self.BLOCK_BASE_WIDTH = 100
        self.BLOCK_BASE_DEPTH = 100
        self.BLOCK_HEIGHT = 20
        self.MOVE_SPEED = 4
        self.WIN_HEIGHT = 20
        self.MAX_STEPS = 1000
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.height = 0
        self.game_over = False
        self.blocks = []
        self.current_block = None
        self.current_block_move_dir = 1
        self.particles = []
        self.rng = None
        
        # Initialize state variables
        self.reset()

        # Run validation check
        # self.validate_implementation()

    def _create_block(self, pos, size, color):
        """Helper to create a block dictionary."""
        return {'pos': list(pos), 'size': list(size), 'color': color}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.height = 0
        self.particles = []

        # Create the initial base block
        base_pos = [self.screen_width / 2, 0, self.screen_width / 2] # x, y (height), z
        base_size = [self.BLOCK_BASE_WIDTH, self.BLOCK_HEIGHT, self.BLOCK_BASE_DEPTH]
        self.blocks = [self._create_block(base_pos, base_size, self.BLOCK_COLORS[0])]
        
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()

    def _spawn_new_block(self):
        last_block = self.blocks[-1]
        new_height_level = self.height + 1
        
        # Calculate landing zone shrink factor based on the next level
        if new_height_level < 5:
            shrink_factor = 1.0
        elif 5 <= new_height_level <= 15:
            # Linear interpolation from 1.0 (at 4) to 0.9 (at 15)
            progress = (new_height_level - 5) / (15 - 5)
            shrink_factor = 0.95 - progress * 0.05
        else: # new_height_level > 15
            shrink_factor = 0.90
        
        new_width = max(20, last_block['size'][0] * shrink_factor)
        new_depth = max(20, last_block['size'][2] * shrink_factor)
        
        # Spawn position: high above, offset to the side
        spawn_y = (self.height + 1) * self.BLOCK_HEIGHT + 150
        
        if self.rng.random() < 0.5:
            spawn_x = 100
            self.current_block_move_dir = 1 # moving right
        else:
            spawn_x = self.screen_width - 100
            self.current_block_move_dir = -1 # moving left
            
        new_pos = [spawn_x, spawn_y, self.screen_width / 2]
        new_size = [new_width, self.BLOCK_HEIGHT, new_depth]
        color_index = new_height_level % len(self.BLOCK_COLORS)
        
        self.current_block = self._create_block(new_pos, new_size, self.BLOCK_COLORS[color_index])
        self.current_block['target_y'] = (self.height + 1) * self.BLOCK_HEIGHT + 60
        self.current_block['state'] = 'intro'

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0.0
        
        movement = action[0]
        space_pressed = action[1] == 1
        
        # --- Update Game Logic ---
        # State machine for the current block
        if self.current_block['state'] == 'intro':
            # Animate block falling into the "ready" position
            self.current_block['pos'][1] = max(self.current_block['target_y'], self.current_block['pos'][1] - 15)
            if self.current_block['pos'][1] == self.current_block['target_y']:
                self.current_block['state'] = 'ready'
        
        elif self.current_block['state'] == 'ready':
            # Automatic side-to-side movement
            self.current_block['pos'][0] += self.current_block_move_dir * self.MOVE_SPEED * 0.5
            if self.current_block['pos'][0] < 80 or self.current_block['pos'][0] > self.screen_width - 80:
                self.current_block_move_dir *= -1
            
            # Player-controlled movement
            if movement == 3: # Left
                self.current_block['pos'][0] -= self.MOVE_SPEED
            elif movement == 4: # Right
                self.current_block['pos'][0] += self.MOVE_SPEED
            
            self.current_block['pos'][0] = np.clip(self.current_block['pos'][0], 0, self.screen_width)

            if space_pressed:
                self.current_block['state'] = 'dropping'
                # sfx: drop_sound.play()
                
        elif self.current_block['state'] == 'dropping':
            target_y = (self.height + 1) * self.BLOCK_HEIGHT
            self.current_block['pos'][1] = max(target_y, self.current_block['pos'][1] - 25)

            if self.current_block['pos'][1] == target_y:
                # --- Landing Check ---
                last_block = self.blocks[-1]
                cb_x, cb_z = self.current_block['pos'][0], self.current_block['pos'][2]
                cb_w, cb_d = self.current_block['size'][0], self.current_block['size'][2]
                
                lb_x, lb_z = last_block['pos'][0], last_block['pos'][2]
                lb_w, lb_d = last_block['size'][0], last_block['size'][2]

                is_success = (abs(cb_x - lb_x) * 2 <= (lb_w - cb_w)) and \
                             (abs(cb_z - lb_z) * 2 <= (lb_d - cb_d))
                
                if is_success:
                    # sfx: success_chime.play()
                    self.height += 1
                    self.blocks.append(self.current_block)
                    
                    reward += 0.1  # Reward for successful placement
                    if self.height > 5:
                        reward += 1.0 # Bonus for higher levels
                    
                    self._create_particles(self.current_block['pos'], self.current_block['color'])
                    
                    if self.height >= self.WIN_HEIGHT:
                        reward += 100.0
                        self.game_over = True
                    else:
                        self._spawn_new_block()
                else:
                    # sfx: failure_sound.play()
                    reward = -100.0
                    self.game_over = True
                    self.current_block['state'] = 'failed'

        self._update_particles()
        self.score += reward
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1

    def _create_particles(self, pos, color):
        # sfx: particle_burst.play()
        for _ in range(25):
            angle = self.rng.random() * math.pi # Erupt upwards
            speed = self.rng.random() * 3 + 1
            self.particles.append({
                'pos': [pos[0], pos[1] + self.BLOCK_HEIGHT],
                'vel': [math.cos(angle + math.pi/2) * speed, -math.sin(angle) * speed * 2],
                'life': self.rng.integers(20, 40),
                'color': color,
                'radius': self.rng.random() * 2 + 1
            })

    def _to_iso(self, x, y, z):
        """Converts 3D world coords to 2D screen coords."""
        iso_x = self.screen_width / 2 + (x - z) * 0.7
        iso_y = 100 + (x + z) * 0.35 - y
        return int(iso_x), int(iso_y)

    def _draw_iso_block(self, block, is_current=False):
        x, y, z = block['pos']
        w, h, d = block['size']
        color = block['color']
        
        corners = [
            (x - w/2, y, z - d/2), (x + w/2, y, z - d/2), (x + w/2, y, z + d/2), (x - w/2, y, z + d/2),
            (x - w/2, y + h, z - d/2), (x + w/2, y + h, z - d/2), (x + w/2, y + h, z + d/2), (x - w/2, y + h, z + d/2)
        ]
        iso_corners = [self._to_iso(cx, cy, cz) for cx, cy, cz in corners]
        
        top_color = color
        side_color1 = color.lerp(pygame.Color('black'), 0.3)
        side_color2 = color.lerp(pygame.Color('black'), 0.5)
        
        # Draw faces with shading
        pygame.gfxdraw.filled_polygon(self.screen, [iso_corners[i] for i in [4, 5, 6, 7]], top_color)
        pygame.gfxdraw.filled_polygon(self.screen, [iso_corners[i] for i in [0, 3, 7, 4]], side_color1)
        pygame.gfxdraw.filled_polygon(self.screen, [iso_corners[i] for i in [3, 2, 6, 7]], side_color2)

        # Draw outlines for clarity
        outline_color = (255, 255, 255, 150) if is_current else (0, 0, 0, 50)
        lines = [(4,5), (5,6), (6,7), (7,4), (0,3), (3,7), (7,4), (2,6), (6,5), (1,5), (0,4), (3,2)]
        visible_lines = set(lines)
        for p1_idx, p2_idx in visible_lines:
             pygame.draw.aaline(self.screen, outline_color, iso_corners[p1_idx], iso_corners[p2_idx])

    def _draw_landing_zone(self, block):
        x, y, z = block['pos']
        w, h, d = block['size']
        corners = [
            (x - w/2, y + h, z - d/2), (x + w/2, y + h, z - d/2),
            (x + w/2, y + h, z + d/2), (x - w/2, y + h, z + d/2)
        ]
        iso_corners = [self._to_iso(cx, cy, cz) for cx, cy, cz in corners]
        pygame.draw.lines(self.screen, self.COLOR_LANDING_ZONE, True, iso_corners, 2)

    def _draw_grid(self):
        for i in range(0, self.screen_width + 200, 50):
            pygame.draw.aaline(self.screen, self.COLOR_GRID, self._to_iso(i, 0, 0), self._to_iso(i, 0, self.screen_width * 2))
            pygame.draw.aaline(self.screen, self.COLOR_GRID, self._to_iso(0, 0, i), self._to_iso(self.screen_width * 2, 0, i))

    def _draw_particles(self):
        for p in self.particles:
            iso_pos = self._to_iso(p['pos'][0], p['pos'][1], self.screen_width / 2)
            alpha = max(0, int(255 * (p['life'] / 40.0)))
            color = (*p['color'][:3], alpha)
            pygame.gfxdraw.filled_circle(self.screen, iso_pos[0], iso_pos[1], int(p['radius']), color)

    def _render_game(self):
        self._draw_grid()
        
        all_blocks = self.blocks
        if self.current_block and self.current_block['state'] != 'failed':
            all_blocks = self.blocks + [self.current_block]

        # Draw blocks from bottom to top
        for block in sorted(all_blocks, key=lambda b: b['pos'][1]):
            is_current = self.current_block and block is self.current_block
            self._draw_iso_block(block, is_current)
        
        if not self.game_over:
            self._draw_landing_zone(self.blocks[-1])
            
        if self.current_block and self.current_block['state'] == 'failed':
            self._draw_iso_block(self.current_block)

        self._draw_particles()

    def _render_ui(self):
        height_text = self.FONT.render(f"Height: {self.height}", True, (255, 255, 255))
        score_text = self.FONT.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(height_text, (20, 20))
        self.screen.blit(score_text, (20, 55))
        
        if self.game_over:
            msg, color = ("YOU WIN!", "#2ecc71") if self.height >= self.WIN_HEIGHT else ("GAME OVER", "#e74c3c")
            end_text = self.FONT.render(msg, True, pygame.Color(color))
            text_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(40, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(end_text, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "height": self.height}
        
    def validate_implementation(self):
        ''' Call this at the end of __init__ to verify implementation. '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Tower Stacker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_r]: # Reset on 'r' key
             obs, info = env.reset()
             total_reward = 0

        action = [movement, space, 0] # Movement, Space, Shift
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Height: {info['height']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset() # auto-reset
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    pygame.quit()