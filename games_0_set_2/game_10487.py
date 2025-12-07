import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:33:57.300510
# Source Brief: brief_00487.md
# Brief Index: 487
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent stacks falling blocks.
    The goal is to build a tall, stable tower. The game ends if the
    tower collapses or if 50 blocks are successfully stacked.
    """
    game_description = (
        "Stack falling blocks to build the tallest tower possible. The game ends if the tower collapses or 50 blocks are successfully stacked."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to move the falling block. Hold space to make the block drop faster."
    )
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GROUND_Y = self.HEIGHT - 40
        self.BLOCK_UNIT_SIZE = 25
        self.MAX_STEPS = 5000
        self.WIN_CONDITION = 50

        # --- Physics Constants ---
        self.PLAYER_SPEED = 5.0
        self.GRAVITY = 0.4
        self.FAST_DROP_MULTIPLIER = 3.0

        # --- Visuals ---
        self.COLOR_BG_TOP = (135, 206, 235) # Sky Blue
        self.COLOR_BG_BOTTOM = (211, 235, 245) # Lighter Sky Blue
        self.COLOR_GROUND = (60, 60, 60)
        self.COLOR_GROUND_BLOCK = (80, 80, 80)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_SHADOW = (0, 0, 0, 128)
        self.BLOCK_DEFINITIONS = [
            # Level 1
            {'size': (2, 1), 'weight': 1.0, 'color': (255, 87, 34)},  # Deep Orange
            # Level 2
            {'size': (3, 1), 'weight': 1.5, 'color': (76, 175, 80)},   # Green
            # Level 3
            {'size': (1, 2), 'weight': 1.0, 'color': (33, 150, 243)},  # Blue
            # Level 4
            {'size': (2, 2), 'weight': 2.0, 'color': (255, 193, 7)},  # Amber
            # Level 5
            {'size': (4, 1), 'weight': 2.0, 'color': (156, 39, 176)}, # Purple
            # Level 6
            {'size': (1, 3), 'weight': 1.5, 'color': (0, 188, 212)},   # Cyan
            # Level 7+
            {'size': (2, 0.5), 'weight': 0.5, 'color': (233, 30, 99)}, # Pink
        ]


        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 64, bold=True)

        # --- State Variables ---
        self.stacked_blocks = []
        self.falling_block = None
        self.particles = []
        self.block_count = 0
        self.level = 1
        self.steps = 0
        self.game_over = False
        self.win = False
        self.wobble = 0.0

        # Initialize state and validate
        self.reset()
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.block_count = 0
        self.level = 1
        self.game_over = False
        self.win = False
        self.wobble = 0.0
        self.particles = []

        # Create a wide, heavy base block on the ground
        base_width = 300
        self.stacked_blocks = [{
            'rect': pygame.Rect(
                (self.WIDTH - base_width) / 2, self.GROUND_Y, base_width, 20
            ),
            'color': self.COLOR_GROUND_BLOCK,
            'weight': 1000, # Effectively immovable
            'wobble_offset': (0, 0),
            'size': (base_width / self.BLOCK_UNIT_SIZE, 20 / self.BLOCK_UNIT_SIZE)
        }]

        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement)

        # --- Game Logic Update ---
        if self.falling_block:
            self._update_falling_block(space_held)
            collision_data = self._check_collision()

            if collision_data['collided']:
                # sound: block_place_thud
                self._handle_block_landing(collision_data['surface_y'])

                reward += 0.1 # Reward for placing a block
                if self.block_count % 5 == 0 and self.block_count > 0:
                    self.level = min((self.block_count // 5) + 1, len(self.BLOCK_DEFINITIONS))
                    reward += 1.0 # Reward for leveling up
                    # sound: level_up_chime

                is_stable = self._check_stability()
                if not is_stable:
                    self.game_over = True
                    reward = -100.0 # Large penalty for losing
                    # sound: tower_crash
                elif self.block_count >= self.WIN_CONDITION:
                    self.game_over = True
                    self.win = True
                    reward = 100.0 # Large bonus for winning
                    # sound: victory_fanfare
                else:
                    self._spawn_new_block()

        self._update_particles()

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not self.game_over:
            terminated = True
            # Optional: small penalty for running out of time
            # reward -= 10

        terminated = self.game_over or terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement):
        if not self.falling_block:
            return

        if movement == 3: # Left
            self.falling_block['rect'].x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.falling_block['rect'].x += self.PLAYER_SPEED

        # Clamp to screen bounds
        self.falling_block['rect'].x = max(0, self.falling_block['rect'].x)
        self.falling_block['rect'].x = min(self.WIDTH - self.falling_block['rect'].width, self.falling_block['rect'].x)

    def _update_falling_block(self, space_held):
        if not self.falling_block:
            return
        
        fall_speed = self.GRAVITY * self.FAST_DROP_MULTIPLIER if space_held else self.GRAVITY
        self.falling_block['y_vel'] += fall_speed
        self.falling_block['rect'].y += self.falling_block['y_vel']

    def _check_collision(self):
        if not self.falling_block:
            return {'collided': False}

        fb_rect = self.falling_block['rect']
        for block in self.stacked_blocks:
            if fb_rect.colliderect(block['rect']):
                # Check if the bottom of the falling block is overlapping the top of a stacked block
                if fb_rect.bottom >= block['rect'].top and self.falling_block['y_vel'] > 0:
                    return {'collided': True, 'surface_y': block['rect'].top}
        return {'collided': False}
    
    def _handle_block_landing(self, surface_y):
        self.falling_block['rect'].bottom = surface_y
        self.stacked_blocks.append(self.falling_block)
        self.block_count += 1
        
        # Create particle effect on impact
        impact_x = self.falling_block['rect'].centerx
        self._create_particles((impact_x, surface_y), self.falling_block['color'], 20)

        self.falling_block = None

    def _spawn_new_block(self):
        # Select a block type based on the current level
        available_types = self.BLOCK_DEFINITIONS[:self.level]
        block_def = random.choice(available_types)

        width = block_def['size'][0] * self.BLOCK_UNIT_SIZE
        height = block_def['size'][1] * self.BLOCK_UNIT_SIZE

        self.falling_block = {
            'rect': pygame.Rect(
                (self.WIDTH - width) / 2, -height, width, height
            ),
            'color': block_def['color'],
            'weight': block_def['weight'],
            'y_vel': 0,
            'wobble_offset': (0, 0),
            'size': block_def['size']
        }

    def _check_stability(self):
        if len(self.stacked_blocks) <= 1:
            return True

        total_mass = 0
        weighted_x_sum = 0
        
        # We only care about the blocks stacked on top of the base
        stack = self.stacked_blocks[1:]

        for block in stack:
            total_mass += block['weight']
            weighted_x_sum += block['rect'].centerx * block['weight']

        if total_mass == 0:
            return True

        center_of_mass_x = weighted_x_sum / total_mass
        
        base_block_rect = self.stacked_blocks[0]['rect']
        support_min_x = base_block_rect.left
        support_max_x = base_block_rect.right
        
        # Calculate wobble factor (-1 to 1)
        support_center = base_block_rect.centerx
        support_width = base_block_rect.width
        self.wobble = (center_of_mass_x - support_center) / (support_width / 2.0)
        self.wobble = np.clip(self.wobble, -1.2, 1.2) # Allow slight over-wobble before topple

        # Topple condition
        if not (support_min_x < center_of_mass_x < support_max_x):
            return False
            
        return True
    
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"block_count": self.block_count, "level": self.level, "steps": self.steps}

    def _render_game(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

        # Draw all blocks
        all_blocks = self.stacked_blocks
        if self.falling_block:
            all_blocks = self.stacked_blocks + [self.falling_block]

        for block in all_blocks:
            self._draw_pretty_rect(self.screen, block['color'], block['rect'], self.wobble, self.GROUND_Y)
            
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 30.0))))
            color = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, color)
            
        self._render_ui()

    def _draw_pretty_rect(self, surface, color, rect, wobble_factor, ground_y):
        # Use a slight wobble effect for blocks in the stack
        is_stacked = rect.bottom >= ground_y

        offset_x = 0
        if is_stacked and rect.height < 21: # Don't wobble the base
            height_factor = (ground_y - rect.centery) / 400.0
            offset_x = wobble_factor * 30.0 * height_factor * height_factor
        
        moved_rect = rect.move(offset_x, 0)
        
        # Darker border color
        border_color = tuple(max(0, c - 40) for c in color)
        
        # Draw main block shape with rounded corners
        pygame.draw.rect(surface, color, moved_rect, border_radius=3)
        # Draw border
        pygame.draw.rect(surface, border_color, moved_rect, width=2, border_radius=3)

    def _render_ui(self):
        # --- Render Text with Shadow ---
        def draw_text(text, font, color, pos):
            shadow_surface = font.render(text, True, self.COLOR_UI_SHADOW)
            text_surface = font.render(text, True, color)
            self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surface, pos)

        # Draw Score (Block Count)
        score_text = f"BLOCKS: {self.block_count}/{self.WIN_CONDITION}"
        draw_text(score_text, self.font_ui, self.COLOR_UI_TEXT, (10, 10))

        # Draw Level
        level_text = f"LEVEL: {self.level}"
        text_width = self.font_ui.size(level_text)[0]
        draw_text(level_text, self.font_ui, self.COLOR_UI_TEXT, (self.WIDTH - text_width - 10, 10))

        # Draw Game Over/Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            text_surf = self.font_game_over.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            shadow_surf = self.font_game_over.render(message, True, (0,0,0))
            self.screen.blit(shadow_surf, text_rect.move(3, 3))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # --- Manual Play Example ---
    # To run this, you need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Stacker")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Action Mapping for Human Player ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Unused in this game
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already the rendered screen, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Resetting ---")
                obs, info = env.reset()
                total_reward = 0
                terminated = False # To restart the loop

        clock.tick(env.metadata["render_fps"])

    print(f"Game Over! Final Score (from rewards): {total_reward:.2f}")
    print(f"Info: {info}")
    
    env.close()