
# Generated: 2025-08-28T04:07:48.899951
# Source Brief: brief_05150.md
# Brief Index: 5150

        
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

    user_guide = (
        "Controls: Arrows to move cursor. Space to place a standard block. "
        "Shift to place a reinforced block. Press Space on the Core to start the wave."
    )

    game_description = (
        "Defend your fortress core against waves of enemies by strategically placing "
        "blocks. Survive 10 waves to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Visuals & Constants ---
        self.GRID_SIZE = 20
        self.COLS = self.SCREEN_WIDTH // self.GRID_SIZE
        self.ROWS = self.SCREEN_HEIGHT // self.GRID_SIZE

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_CORE = (0, 150, 255)
        self.COLOR_CORE_GLOW = (0, 150, 255, 50)
        self.COLOR_BLOCK = (120, 120, 140)
        self.COLOR_REINFORCED_BLOCK = (60, 60, 70)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 50, 50, 50)
        self.COLOR_CURSOR_VALID = (255, 255, 255)
        self.COLOR_CURSOR_INVALID = (255, 0, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_HIT = (255, 255, 255)

        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 36, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.game_phase = None
        self.wave = None
        self.blocks_to_place = None
        self.message_timer = None

        self.core = None
        self.cursor = None
        self.blocks = None
        self.enemies = None
        self.particles = None

        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.game_phase = "PLACEMENT"
        self.wave = 1
        self.blocks_to_place = 10
        self.message_timer = 60 # Show "Prepare" message for 2s

        core_size = self.GRID_SIZE * 2
        self.core = {
            'rect': pygame.Rect(
                self.SCREEN_WIDTH // 2 - core_size // 2,
                self.SCREEN_HEIGHT // 2 - core_size // 2,
                core_size, core_size
            ),
            'health': 100,
            'max_health': 100,
            'hit_timer': 0
        }

        self.cursor = {
            'grid_pos': [self.COLS // 2, self.ROWS // 2 - 3],
            'valid_placement': True
        }
        
        self.blocks = []
        self.enemies = []
        self.particles = []

        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        if self.game_over or self.game_won:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        if self.message_timer > 0:
            self.message_timer -= 1

        if self.game_phase == "PLACEMENT":
            self._handle_placement_phase(movement, space_press, shift_press)
        elif self.game_phase == "COMBAT":
            reward += self._handle_combat_phase()

        self._update_particles()
        
        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and self.game_over:
            reward -= 100.0
        elif terminated and self.game_won:
            reward += 100.0

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_placement_phase(self, movement, space_press, shift_press):
        self._move_cursor(movement)
        self._update_cursor_validity()
        
        cursor_rect = self._get_cursor_rect()
        
        if space_press and cursor_rect.colliderect(self.core['rect']):
            self._start_wave()
            return

        if self.cursor['valid_placement']:
            if space_press:
                self._place_block(reinforced=False)
            elif shift_press:
                self._place_block(reinforced=True)

    def _handle_combat_phase(self):
        reward = 0.0
        reward += self._update_enemies()
        reward += self._check_collisions()

        if not self.enemies and not self.game_over:
            reward += 1.0  # Wave survived
            self.wave += 1
            if self.wave > 10:
                self.game_won = True
            else:
                self.game_phase = "PLACEMENT"
                self.blocks_to_place += 5 + self.wave
                self.message_timer = 60 # Show "Wave Complete"
        return reward

    def _move_cursor(self, movement):
        if movement == 1: self.cursor['grid_pos'][1] -= 1  # Up
        elif movement == 2: self.cursor['grid_pos'][1] += 1  # Down
        elif movement == 3: self.cursor['grid_pos'][0] -= 1  # Left
        elif movement == 4: self.cursor['grid_pos'][0] += 1  # Right
        self.cursor['grid_pos'][0] = np.clip(self.cursor['grid_pos'][0], 0, self.COLS - 1)
        self.cursor['grid_pos'][1] = np.clip(self.cursor['grid_pos'][1], 0, self.ROWS - 1)

    def _get_cursor_rect(self):
        return pygame.Rect(
            self.cursor['grid_pos'][0] * self.GRID_SIZE,
            self.cursor['grid_pos'][1] * self.GRID_SIZE,
            self.GRID_SIZE, self.GRID_SIZE
        )

    def _update_cursor_validity(self):
        cursor_rect = self._get_cursor_rect()
        if cursor_rect.colliderect(self.core['rect']):
            self.cursor['valid_placement'] = False
            return
        for block in self.blocks:
            if cursor_rect.colliderect(block['rect']):
                self.cursor['valid_placement'] = False
                return
        self.cursor['valid_placement'] = True

    def _place_block(self, reinforced: bool):
        cost = 2 if reinforced else 1
        if self.blocks_to_place < cost:
            # sfx: error sound
            return

        self.blocks_to_place -= cost
        health = 10 if reinforced else 3
        block_type = "reinforced" if reinforced else "standard"
        
        self.blocks.append({
            'rect': self._get_cursor_rect(),
            'health': health,
            'max_health': health,
            'type': block_type,
            'hit_timer': 0
        })
        # sfx: place block
        self._update_cursor_validity()

    def _start_wave(self):
        self.game_phase = "COMBAT"
        num_enemies = 2 + self.wave
        enemy_speed = 0.5 + self.wave * 0.1
        enemy_health = 1 + self.wave
        
        for _ in range(num_enemies):
            side = self.np_random.integers(4)
            x, y = 0, 0
            if side == 0:  # Top
                x = self.np_random.uniform(0, self.SCREEN_WIDTH)
                y = -self.GRID_SIZE
            elif side == 1:  # Bottom
                x = self.np_random.uniform(0, self.SCREEN_WIDTH)
                y = self.SCREEN_HEIGHT
            elif side == 2:  # Left
                x = -self.GRID_SIZE
                y = self.np_random.uniform(0, self.SCREEN_HEIGHT)
            else:  # Right
                x = self.SCREEN_WIDTH
                y = self.np_random.uniform(0, self.SCREEN_HEIGHT)

            self.enemies.append({
                'pos': np.array([x, y], dtype=float),
                'health': enemy_health,
                'max_health': enemy_health,
                'speed': enemy_speed,
                'hit_timer': 0,
                'rect': pygame.Rect(x, y, self.GRID_SIZE, self.GRID_SIZE)
            })
        # sfx: wave start horn

    def _update_enemies(self):
        reward = 0.0
        core_center = np.array(self.core['rect'].center, dtype=float)
        
        for enemy in self.enemies[:]:
            direction = core_center - enemy['pos']
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction /= dist

            # Simple pathfinding
            proposed_pos = enemy['pos'] + direction * enemy['speed']
            
            # Check for block collisions
            is_blocked = False
            for block in self.blocks:
                # Create a slightly larger rect for collision prediction
                predict_rect = pygame.Rect(proposed_pos[0], proposed_pos[1], self.GRID_SIZE, self.GRID_SIZE)
                if predict_rect.colliderect(block['rect']):
                    is_blocked = True
                    block['health'] -= 1
                    block['hit_timer'] = 5
                    if block['health'] <= 0:
                        self._create_explosion(block['rect'].center, self.COLOR_BLOCK, 10)
                        self.blocks.remove(block)
                        reward -= 0.01 # Penalty for losing a block
                        # sfx: block break
                    break
            
            if is_blocked:
                # sfx: enemy hit block
                self._create_explosion(enemy['pos'], self.COLOR_ENEMY, 3)
                enemy['health'] -= 1
                enemy['hit_timer'] = 5
                if enemy['health'] <= 0:
                    self._create_explosion(enemy['pos'], self.COLOR_ENEMY, 20)
                    self.enemies.remove(enemy)
                    reward += 0.1 # Reward for destroying enemy
                    # sfx: enemy explode
            else:
                enemy['pos'] = proposed_pos
                enemy['rect'].topleft = enemy['pos']
        
        return reward

    def _check_collisions(self):
        reward = 0.0
        # Enemy-Core collision
        for enemy in self.enemies[:]:
            if enemy['rect'].colliderect(self.core['rect']):
                self.core['health'] -= 5
                self.core['hit_timer'] = 10
                self._create_explosion(enemy['rect'].center, self.COLOR_ENEMY, 20)
                self.enemies.remove(enemy)
                reward += 0.1 # Reward for destroying enemy (even by suicide)
                # sfx: core hit, enemy explode
                if self.core['health'] <= 0:
                    self.game_over = True
                    self._create_explosion(self.core['rect'].center, self.COLOR_CORE, 100)
                    # sfx: game over
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': vel,
                'life': self.np_random.integers(10, 30),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _check_termination(self):
        if self.game_over or self.game_won:
            return True
        if self.steps >= 10000:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw blocks
        for block in self.blocks:
            color = self.COLOR_HIT if block['hit_timer'] > 0 else (self.COLOR_REINFORCED_BLOCK if block['type'] == 'reinforced' else self.COLOR_BLOCK)
            pygame.draw.rect(self.screen, color, block['rect'])
            if block['hit_timer'] > 0: block['hit_timer'] -= 1

        # Draw core
        if self.core['hit_timer'] > 0:
            pygame.draw.rect(self.screen, self.COLOR_HIT, self.core['rect'])
            self.core['hit_timer'] -= 1
        else:
            pygame.gfxdraw.box(self.screen, self.core['rect'], self.COLOR_CORE)
        glow_radius = int(self.core['rect'].width * 0.75)
        pygame.gfxdraw.filled_circle(self.screen, self.core['rect'].centerx, self.core['rect'].centery, glow_radius, self.COLOR_CORE_GLOW)
        
        # Draw enemies
        for enemy in self.enemies:
            color = self.COLOR_HIT if enemy['hit_timer'] > 0 else self.COLOR_ENEMY
            pygame.gfxdraw.box(self.screen, enemy['rect'], color)
            if enemy['hit_timer'] > 0: enemy['hit_timer'] -= 1
            pygame.gfxdraw.filled_circle(self.screen, enemy['rect'].centerx, enemy['rect'].centery, self.GRID_SIZE, self.COLOR_ENEMY_GLOW)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            color_with_alpha = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color_with_alpha)

        # Draw cursor in placement phase
        if self.game_phase == "PLACEMENT":
            rect = self._get_cursor_rect()
            color = self.COLOR_CURSOR_VALID if self.cursor['valid_placement'] else self.COLOR_CURSOR_INVALID
            pygame.draw.rect(self.screen, color, rect, 2)

    def _render_ui(self):
        # Wave counter
        wave_text = self.font_ui.render(f"WAVE: {self.wave}/10", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Blocks counter
        block_text = self.font_ui.render(f"BLOCKS: {self.blocks_to_place}", True, self.COLOR_TEXT)
        self.screen.blit(block_text, (self.SCREEN_WIDTH - block_text.get_width() - 10, 10))

        # Core health bar
        health_frac = max(0, self.core['health'] / self.core['max_health'])
        bar_width = self.core['rect'].width
        bar_height = 8
        bar_x = self.core['rect'].left
        bar_y = self.core['rect'].bottom + 5
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_CORE, (bar_x, bar_y, int(bar_width * health_frac), bar_height))

        # Phase/Game Over messages
        msg_text = None
        if self.game_over:
            msg_text = self.font_msg.render("FORTRESS LOST", True, self.COLOR_ENEMY)
        elif self.game_won:
            msg_text = self.font_msg.render("VICTORY!", True, self.COLOR_CORE)
        elif self.message_timer > 0:
            if self.game_phase == "PLACEMENT" and self.wave > 1:
                msg_text = self.font_msg.render("WAVE COMPLETE", True, self.COLOR_TEXT)
            elif self.game_phase == "PLACEMENT":
                msg_text = self.font_msg.render(f"PREPARE FOR WAVE {self.wave}", True, self.COLOR_TEXT)
        
        if msg_text:
            pos = (self.SCREEN_WIDTH // 2 - msg_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg_text.get_height() // 2)
            self.screen.blit(msg_text, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "core_health": self.core['health'],
            "blocks_remaining": self.blocks_to_place,
            "game_phase": self.game_phase,
        }

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Requires `pip install pygame`
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a dictionary to track held keys for smoother controls
    keys_held = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
        pygame.K_RSHIFT: False,
    }
    
    # Create a window to display the game
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Fortress")
    
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    
    while not done:
        # --- Action mapping from keyboard to MultiDiscrete action space ---
        movement = 0 # No-op
        if keys_held[pygame.K_UP]: movement = 1
        elif keys_held[pygame.K_DOWN]: movement = 2
        elif keys_held[pygame.K_LEFT]: movement = 3
        elif keys_held[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys_held[pygame.K_SPACE] else 0
        shift = 1 if (keys_held[pygame.K_LSHIFT] or keys_held[pygame.K_RSHIFT]) else 0
        
        action = [movement, space, shift]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
                if event.key == pygame.K_r: # Reset on 'R' key
                    obs, info = env.reset()
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        # --- Render the observation to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        clock.tick(30)
        
    env.close()
    print("Game Over. Final Info:", info)