
# Generated: 2025-08-27T22:10:28.470879
# Source Brief: brief_03038.md
# Brief Index: 3038

        
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
        "Controls: Use arrow keys to position the block. Press space to drop it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack blocks to reach the target height. You have a limited number of blocks, so place them carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.TARGET_HEIGHT = 10
        self.TOTAL_BLOCKS = 20
        self.MAX_STEPS = 1000
        self.GRID_SIZE = 7 # 7x7 grid for placement
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Colors
        self.COLOR_BG_TOP = pygame.Color("#2c3e50")
        self.COLOR_BG_BOTTOM = pygame.Color("#1a2530")
        self.COLOR_GRID = pygame.Color(100, 100, 120, 50)
        self.COLOR_TARGET_LINE = pygame.Color("#2ecc71")
        self.COLOR_FALLING_BLOCK = pygame.Color("#e74c3c")
        self.COLOR_FALLING_BLOCK_SHADOW = pygame.Color("#c0392b")
        self.COLOR_TEXT = pygame.Color(230, 230, 230)
        self.COLOR_TEXT_SHADOW = pygame.Color(20, 20, 20)
        self.BLUE_HUE = 200

        # Isometric projection constants
        self.ISO_BLOCK_WIDTH = 32
        self.ISO_BLOCK_HEIGHT = 32
        self.ISO_Z_HEIGHT = self.ISO_BLOCK_HEIGHT // 2
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = self.HEIGHT - 50

        # State variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.placed_blocks = []
        self.blocks_remaining = 0
        self.particles = []
        self.falling_block = None
        self.current_stack_height = 0
        self.last_space_held = False
        self.bg_anim_offset = 0

        # Initialize state
        self.reset()
        
        # Run self-validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.placed_blocks = []
        self.blocks_remaining = self.TOTAL_BLOCKS
        self.particles = []
        self.current_stack_height = 0
        self.last_space_held = False

        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        # --- Game Logic ---
        
        # 1. Handle player input
        if self.falling_block:
            grid_limit = self.GRID_SIZE // 2
            if movement == 1: # Up
                self.falling_block['y'] = max(-grid_limit, self.falling_block['y'] - 1)
            elif movement == 2: # Down
                self.falling_block['y'] = min(grid_limit, self.falling_block['y'] + 1)
            elif movement == 3: # Left
                self.falling_block['x'] = max(-grid_limit, self.falling_block['x'] - 1)
            elif movement == 4: # Right
                self.falling_block['x'] = min(grid_limit, self.falling_block['x'] + 1)

            # 2. Handle block drop
            is_drop_action = space_pressed
            if is_drop_action:
                self.falling_block['z'] = self._get_height_at(self.falling_block['x'], self.falling_block['y'])
            else:
                self.falling_block['z'] -= 0.2 # Slow continuous fall

            # 3. Check for landing
            landing_height = self._get_height_at(self.falling_block['x'], self.falling_block['y'])
            if self.falling_block['z'] <= landing_height:
                # Block has landed
                # # Sound: block_place.wav
                self.falling_block['z'] = landing_height
                
                # Add to placed blocks with a color based on height
                lightness = 50 + landing_height * 3
                color = pygame.Color(0, 0, 0)
                color.hsla = (self.BLUE_HUE, 100, max(20, min(80, lightness)), 100)
                self.placed_blocks.append({**self.falling_block, 'color': color})
                
                # Spawn particles
                self._create_particles(self.falling_block['x'], self.falling_block['y'], landing_height, color)

                # Calculate reward for height gain
                prev_height = self.current_stack_height
                self.current_stack_height = self._calculate_stack_height()
                height_gain = self.current_stack_height - prev_height
                if height_gain > 0:
                    reward += height_gain * 0.1

                # Spawn next block
                self._spawn_new_block()

        # 4. Update animations
        self._update_particles()
        self.bg_anim_offset = (self.bg_anim_offset + 0.5) % self.WIDTH

        self.steps += 1
        
        # 5. Check for termination
        terminated = self._check_termination()
        if terminated:
            if self.win:
                win_reward = 5.0
                if self.blocks_remaining > 0:
                    win_reward += 10.0 # Bonus for efficiency
                reward += win_reward
            elif self.blocks_remaining <= 0 and not self.win:
                reward -= 10.0 # Penalty for failure
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_new_block(self):
        self.blocks_remaining -= 1
        if self.blocks_remaining >= 0:
            self.falling_block = {
                'x': 0, 
                'y': 0, 
                'z': self.TARGET_HEIGHT + 4,
                'color': self.COLOR_FALLING_BLOCK
            }
        else:
            self.falling_block = None

    def _get_height_at(self, x, y):
        max_h = 0
        for block in self.placed_blocks:
            if block['x'] == x and block['y'] == y:
                max_h = max(max_h, block['z'] + 1)
        return max_h

    def _calculate_stack_height(self):
        if not self.placed_blocks:
            return 0
        return max(block['z'] + 1 for block in self.placed_blocks)

    def _check_termination(self):
        if self.game_over:
            return True
        
        if self.current_stack_height >= self.TARGET_HEIGHT:
            self.win = True
            self.game_over = True
            return True
        
        if self.blocks_remaining < 0 and self.falling_block is None:
            self.game_over = True
            return True
            
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

    def _to_iso(self, x, y, z):
        """Converts 3D grid coordinates to 2D screen coordinates."""
        screen_x = self.ORIGIN_X + (x - y) * (self.ISO_BLOCK_WIDTH / 2)
        screen_y = self.ORIGIN_Y + (x + y) * (self.ISO_BLOCK_HEIGHT / 4) - z * self.ISO_Z_HEIGHT
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, x, y, z, color, shadow_color=None):
        """Draws a 3D isometric cube."""
        if shadow_color is None:
            shadow_color = color.lerp((0,0,0), 0.3)
        side_color = color.lerp((0,0,0), 0.15)
        
        w, h, zh = self.ISO_BLOCK_WIDTH, self.ISO_BLOCK_HEIGHT, self.ISO_Z_HEIGHT
        
        # Calculate 8 corners of the cube in screen space
        p = [self._to_iso(x, y, z),
             self._to_iso(x + 1, y, z),
             self._to_iso(x + 1, y + 1, z),
             self._to_iso(x, y + 1, z),
             self._to_iso(x, y, z + 1),
             self._to_iso(x + 1, y, z + 1),
             self._to_iso(x + 1, y + 1, z + 1),
             self._to_iso(x, y + 1, z + 1)]

        # Draw faces with anti-aliasing
        pygame.gfxdraw.filled_polygon(surface, (p[0], p[1], p[5], p[4]), shadow_color) # Right face
        pygame.gfxdraw.aapolygon(surface, (p[0], p[1], p[5], p[4]), shadow_color)
        
        pygame.gfxdraw.filled_polygon(surface, (p[0], p[3], p[7], p[4]), side_color) # Left face
        pygame.gfxdraw.aapolygon(surface, (p[0], p[3], p[7], p[4]), side_color)
        
        pygame.gfxdraw.filled_polygon(surface, (p[4], p[5], p[6], p[7]), color) # Top face
        pygame.gfxdraw.aapolygon(surface, (p[4], p[5], p[6], p[7]), color)

    def _create_particles(self, x, y, z, color):
        px, py = self._to_iso(x + 0.5, y + 0.5, z)
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 1],
                'life': random.randint(20, 40),
                'color': color,
                'size': random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # 1. Draw background
        for i in range(self.HEIGHT):
            ratio = i / self.HEIGHT
            color = self.COLOR_BG_TOP.lerp(self.COLOR_BG_BOTTOM, ratio)
            self.screen.fill(color, (0, i, self.WIDTH, 1))

        # Subtle background animation
        temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for i in range(0, self.WIDTH, 20):
            x = (i + self.bg_anim_offset) % self.WIDTH
            pygame.draw.line(temp_surf, (255, 255, 255, 5), (x, 0), (x - 40, self.HEIGHT), 2)
        self.screen.blit(temp_surf, (0,0))
        
        # 2. Render all game elements
        self._render_game()
        
        # 3. Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            # Lines along X
            p1 = self._to_iso(-self.GRID_SIZE/2, -self.GRID_SIZE/2 + i, 0)
            p2 = self._to_iso(self.GRID_SIZE/2, -self.GRID_SIZE/2 + i, 0)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
            # Lines along Y
            p1 = self._to_iso(-self.GRID_SIZE/2 + i, -self.GRID_SIZE/2, 0)
            p2 = self._to_iso(-self.GRID_SIZE/2 + i, self.GRID_SIZE/2, 0)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)

        # Draw target height line
        p1 = self._to_iso(-self.GRID_SIZE/2-0.5, self.GRID_SIZE/2+0.5, self.TARGET_HEIGHT)
        p2 = self._to_iso(self.GRID_SIZE/2+0.5, self.GRID_SIZE/2+0.5, self.TARGET_HEIGHT)
        p3 = self._to_iso(self.GRID_SIZE/2+0.5, -self.GRID_SIZE/2-0.5, self.TARGET_HEIGHT)
        pygame.draw.aaline(self.screen, self.COLOR_TARGET_LINE, p1, p2, 2)
        pygame.draw.aaline(self.screen, self.COLOR_TARGET_LINE, p2, p3, 2)

        # Sort all drawable items for painter's algorithm
        drawable_items = []
        if self.falling_block:
            drawable_items.append(('block', self.falling_block))
            # Shadow for falling block
            shadow_z = self._get_height_at(self.falling_block['x'], self.falling_block['y'])
            drawable_items.append(('shadow', {**self.falling_block, 'z': shadow_z}))
        
        for block in self.placed_blocks:
            drawable_items.append(('block', block))

        # Sort by depth: lower on screen = drawn later
        drawable_items.sort(key=lambda item: (item[1]['x'] + item[1]['y']) * 0.5 + item[1]['z'])

        for item_type, item_data in drawable_items:
            if item_type == 'block':
                self._draw_iso_cube(self.screen, item_data['x'], item_data['y'], item_data['z'], item_data['color'])
            elif item_type == 'shadow':
                 # Draw a simple projection on the landing spot
                color = item_data['color'].lerp((0,0,0,0), 0.5)
                color.a = 100
                self._draw_iso_cube(self.screen, item_data['x'], item_data['y'], item_data['z'], color, color)


        # Draw particles on top
        for p in self.particles:
            alpha = p['life'] * 6
            color = p['color']
            temp_color = pygame.Color(color.r, color.g, color.b, min(255, int(alpha)))
            pygame.draw.circle(self.screen, temp_color, p['pos'], int(p['size'] * p['life']/40))

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, shadow_color, center=False):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            text_rect = text_surf.get_rect()
            if center:
                text_rect.center = pos
            else:
                text_rect.topleft = pos
            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
            self.screen.blit(text_surf, text_rect)

        # Score and stats
        draw_text(f"Height: {self.current_stack_height}/{self.TARGET_HEIGHT}", self.font_medium, self.COLOR_TEXT, (20, 20), self.COLOR_TEXT_SHADOW)
        draw_text(f"Score: {self.score:.1f}", self.font_small, self.COLOR_TEXT, (20, 55), self.COLOR_TEXT_SHADOW)
        
        # Remaining blocks icons
        draw_text("Blocks:", self.font_medium, self.COLOR_TEXT, (self.WIDTH - 180, 20), self.COLOR_TEXT_SHADOW)
        for i in range(min(self.blocks_remaining + 1, 10)):
             pygame.draw.rect(self.screen, self.COLOR_FALLING_BLOCK, (self.WIDTH - 170 + i * 14, 55, 10, 10))
             pygame.draw.rect(self.screen, self.COLOR_FALLING_BLOCK_SHADOW, (self.WIDTH - 170 + i * 14, 55, 10, 10), 1)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_TARGET_LINE if self.win else self.COLOR_FALLING_BLOCK
            draw_text(msg, self.font_large, color, (self.WIDTH // 2, self.HEIGHT // 2), self.COLOR_TEXT_SHADOW, center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.current_stack_height,
            "blocks_remaining": self.blocks_remaining
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Stacker")
    
    done = False
    clock = pygame.time.Clock()
    
    # Game loop
    while not done:
        # Action defaults
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Keyboard controls
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Height: {info['height']}")
            # Allow a moment to see the final screen before resetting
            frame = np.transpose(obs, (1, 0, 2))
            pygame.surfarray.blit_array(screen, frame)
            pygame.display.flip()
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(screen, frame)
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for playability
        
    env.close()