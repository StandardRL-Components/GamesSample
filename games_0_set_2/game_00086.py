
# Generated: 2025-08-27T12:33:53.259270
# Source Brief: brief_00086.md
# Brief Index: 86

        
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
        "Controls: ←→ to move block, Shift to cycle block type, Space to drop."
    )

    game_description = (
        "Crush descending monsters with falling blocks in a fast-paced, grid-based action puzzle game."
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
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game Constants ---
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.GRID_OFFSET_X, self.GRID_OFFSET_Y = 170, 0
        self.CELL_SIZE = 20
        self.MAX_STEPS = 5000

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_LANDED = (80, 90, 110)
        self.COLOR_PLAYER_BLOCK = (0, 255, 128)
        self.COLOR_GHOST_BLOCK = (0, 255, 128, 100)
        self.COLOR_MONSTER_1 = (255, 80, 80)
        self.COLOR_MONSTER_2 = (80, 120, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_SCORE = (255, 215, 0)
        
        # --- Block Shapes (relative coordinates) ---
        self.BLOCK_SHAPES = {
            0: [(0, 0), (1, 0), (0, 1), (1, 1)],  # Square 2x2
            1: [(0, 0), (1, 0), (2, 0), (3, 0)],  # Line 4x1
            2: [(0, 0), (0, 1), (1, 1), (2, 1)],  # L-shape
        }
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = None
        self.monsters = []
        self.current_block = None
        self.next_block_type = 0
        self.monster_descent_period = 5.0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.combo_count = 0
        self.particles = []
        self.floating_texts = []
        self.screen_shake = 0
        
        self.reset()
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.monsters = []
        self.monster_descent_period = 50.0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.combo_count = 0
        self.particles = []
        self.floating_texts = []
        self.screen_shake = 0

        self._spawn_monsters(15)
        self.next_block_type = self.np_random.integers(0, len(self.BLOCK_SHAPES))
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Unpack Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Player Input ---
        if self.current_block:
            if movement == 3:  # Left
                self.current_block['x'] -= 1
            elif movement == 4:  # Right
                self.current_block['x'] += 1
            self._clamp_block_position()

            if shift_held and not self.prev_shift_held:
                self._spawn_new_block(cycle=True) # sound: block_cycle.wav
            
            if space_held and not self.prev_space_held:
                reward += self._handle_block_drop() # sound: block_drop.wav
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game World (auto_advance) ---
        self._update_monsters()
        self._update_effects()
        
        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 50 == 0:
            self.monster_descent_period = max(10.0, self.monster_descent_period * 0.98)

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if not self.monsters: # Victory
                reward += 100
                self.floating_texts.append(self._create_floating_text("YOU WIN!", self.WIDTH // 2, self.HEIGHT // 2, size=2, duration=120, color=(0, 255, 0)))
            else: # Loss
                reward -= 100
                self.floating_texts.append(self._create_floating_text("GAME OVER", self.WIDTH // 2, self.HEIGHT // 2, size=2, duration=120, color=(255, 0, 0)))

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_block_drop(self):
        if not self.current_block: return 0

        reward = 0
        landing_y = self._get_landing_y()
        shape_coords = self.BLOCK_SHAPES[self.current_block['type']]
        
        crushed_monsters = []
        for dx, dy in shape_coords:
            x = self.current_block['x'] + dx
            y = landing_y + dy
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                self.grid[x, y] = 3 # Mark as landed block
                
                # Check for monster collision
                for monster in self.monsters:
                    if monster['x'] == x and monster['y'] == y and monster not in crushed_monsters:
                        crushed_monsters.append(monster)
        
        if crushed_monsters:
            # sound: explosion.wav
            self.combo_count += len(crushed_monsters)
            for monster in crushed_monsters:
                self.monsters.remove(monster)
                reward += 1  # +1 per monster
                self._create_explosion(monster['x'], monster['y'])
            
            # Combo reward
            if len(crushed_monsters) > 1:
                reward += 5 # Combo initiation bonus
                reward += (len(crushed_monsters) - 1) # Additional per monster in combo
                self.floating_texts.append(self._create_floating_text(f"x{len(crushed_monsters)} COMBO!", (self.current_block['x'] + 1) * self.CELL_SIZE + self.GRID_OFFSET_X, landing_y * self.CELL_SIZE, color=(255, 165, 0)))
            
            self.floating_texts.append(self._create_floating_text(f"+{int(reward)}", (self.current_block['x'] + 1) * self.CELL_SIZE + self.GRID_OFFSET_X, (landing_y - 1) * self.CELL_SIZE, color=self.COLOR_SCORE))
            self.score += int(reward) # Use the reward as score for simplicity
            self.screen_shake = 10
        else:
            reward = -0.2 # Penalty for safe drop
            self.combo_count = 0
        
        self.current_block = None
        self._spawn_new_block()
        return reward
    
    def _update_monsters(self):
        descent_due = self.steps > 0 and self.steps % int(self.monster_descent_period) == 0
        
        for monster in self.monsters:
            monster['move_counter'] += 1
            # Type 1: Horizontal move every 3 steps
            if monster['type'] == 1 and monster['move_counter'] % 3 == 0:
                nx = monster['x'] + monster['dir']
                if not (0 <= nx < self.GRID_WIDTH) or self.grid[nx, monster['y']] != 0:
                    monster['dir'] *= -1
                else:
                    monster['x'] = nx
            # Type 2: Diagonal move every 2 steps
            elif monster['type'] == 2 and monster['move_counter'] % 2 == 0:
                nx = monster['x'] + monster['dir']
                if not (0 <= nx < self.GRID_WIDTH) or self.grid[nx, monster['y']] != 0:
                    monster['dir'] *= -1
                else:
                    monster['x'] = nx
            
            if descent_due:
                monster['y'] += 1

            if monster['y'] >= self.GRID_HEIGHT - 1:
                self.game_over = True

    def _update_effects(self):
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1
        
        # Update floating texts
        self.floating_texts = [t for t in self.floating_texts if t['life'] > 0]
        for t in self.floating_texts:
            t['y'] -= t['vy']
            t['life'] -= 1
            t['alpha'] = max(0, 255 * (t['life'] / t['duration']))

        # Update screen shake
        if self.screen_shake > 0:
            self.screen_shake -= 1

    def _spawn_monsters(self, count):
        for _ in range(count):
            m_type = self.np_random.integers(1, 3)
            self.monsters.append({
                'x': self.np_random.integers(0, self.GRID_WIDTH),
                'y': self.np_random.integers(0, self.GRID_HEIGHT // 2),
                'type': m_type,
                'move_counter': 0,
                'dir': self.np_random.choice([-1, 1]),
                'anim_offset': self.np_random.random() * 2 * math.pi
            })

    def _spawn_new_block(self, cycle=False):
        if cycle:
            self.current_block['type'] = (self.current_block['type'] + 1) % len(self.BLOCK_SHAPES)
        else:
            block_type = self.next_block_type
            self.current_block = {
                'type': block_type,
                'x': self.GRID_WIDTH // 2 - 1,
                'y': 0,
            }
            self.next_block_type = self.np_random.integers(0, len(self.BLOCK_SHAPES))
        self._clamp_block_position()
    
    def _clamp_block_position(self):
        shape_coords = self.BLOCK_SHAPES[self.current_block['type']]
        min_x = -min(c[0] for c in shape_coords)
        max_x = self.GRID_WIDTH - 1 - max(c[0] for c in shape_coords)
        self.current_block['x'] = max(min_x, min(self.current_block['x'], max_x))

    def _get_landing_y(self):
        if not self.current_block: return self.GRID_HEIGHT - 1
        shape_coords = self.BLOCK_SHAPES[self.current_block['type']]
        x = self.current_block['x']
        
        # Start from top and check downwards
        for y_offset in range(self.GRID_HEIGHT):
            for dx, dy in shape_coords:
                check_x, check_y = x + dx, y_offset + dy
                # Check for collision with landed blocks or floor
                if check_y >= self.GRID_HEIGHT or (0 <= check_x < self.GRID_WIDTH and self.grid[check_x, check_y] != 0):
                    # Collision detected, so it must land on the row above
                    return y_offset - 1
        return self.GRID_HEIGHT - 1 - max(c[1] for c in shape_coords)

    def _check_termination(self):
        if self.game_over:
            return True
        if not self.monsters: # Victory
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Screen Shake ---
        shake_x, shake_y = 0, 0
        if self.screen_shake > 0:
            shake_x = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            shake_y = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)

        # --- Grid ---
        grid_surface = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.HEIGHT))
        grid_surface.set_colorkey((0,0,0))
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                # Landed blocks
                if self.grid[x, y] == 3:
                    pygame.draw.rect(grid_surface, self.COLOR_LANDED, rect)
                # Grid lines
                pygame.draw.rect(grid_surface, self.COLOR_GRID, rect, 1)
        
        # --- Monsters ---
        for monster in self.monsters:
            anim_y = math.sin(self.steps * 0.2 + monster['anim_offset']) * 2
            color = self.COLOR_MONSTER_1 if monster['type'] == 1 else self.COLOR_MONSTER_2
            rect = (monster['x'] * self.CELL_SIZE + 2, monster['y'] * self.CELL_SIZE + 2 + anim_y, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
            pygame.gfxdraw.box(grid_surface, rect, color)
            
        # --- Ghost Block ---
        if self.current_block:
            landing_y = self._get_landing_y()
            shape_coords = self.BLOCK_SHAPES[self.current_block['type']]
            for dx, dy in shape_coords:
                rect = ((self.current_block['x'] + dx) * self.CELL_SIZE, (landing_y + dy) * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.gfxdraw.box(grid_surface, rect, self.COLOR_GHOST_BLOCK)

        # --- Current Block ---
        if self.current_block:
            shape_coords = self.BLOCK_SHAPES[self.current_block['type']]
            for dx, dy in shape_coords:
                rect = ((self.current_block['x'] + dx) * self.CELL_SIZE, (self.current_block['y'] + dy) * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.gfxdraw.box(grid_surface, rect, self.COLOR_PLAYER_BLOCK)
                pygame.draw.rect(grid_surface, tuple(c*0.7 for c in self.COLOR_PLAYER_BLOCK), rect, 2)
        
        self.screen.blit(grid_surface, (self.GRID_OFFSET_X + shake_x, self.GRID_OFFSET_Y + shake_y))

        # --- Particles & Texts ---
        for p in self.particles:
            size = max(0, int(p['size'] * (p['life'] / p['duration'])))
            pygame.draw.rect(self.screen, p['color'], (p['x'] + shake_x, p['y'] + shake_y, size, size))
        
        for t in self.floating_texts:
            font = self.font_large if t['size'] == 2 else self.font_small
            text_surf = font.render(t['text'], True, t['color'])
            text_surf.set_alpha(t['alpha'])
            text_rect = text_surf.get_rect(center=(t['x'] + shake_x, t['y'] + shake_y))
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # --- Score Display ---
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # --- Combo Display ---
        if self.combo_count > 1:
            combo_text = self.font_small.render(f"COMBO: x{self.combo_count}", True, self.COLOR_SCORE)
            self.screen.blit(combo_text, (10, 30))
        
        # --- Monsters Remaining ---
        monster_text = self.font_small.render(f"MONSTERS: {len(self.monsters)}", True, self.COLOR_TEXT)
        self.screen.blit(monster_text, (10, 50))
        
        # --- Next Block Preview ---
        next_text = self.font_small.render("NEXT:", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.WIDTH - 100, 10))
        shape_coords = self.BLOCK_SHAPES[self.next_block_type]
        for dx, dy in shape_coords:
            rect = (self.WIDTH - 100 + dx*self.CELL_SIZE, 40 + dy*self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.gfxdraw.box(self.screen, rect, self.COLOR_PLAYER_BLOCK)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.COLOR_PLAYER_BLOCK), rect, 2)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "monsters_remaining": len(self.monsters),
            "combo": self.combo_count
        }

    def _create_explosion(self, grid_x, grid_y):
        cx = self.GRID_OFFSET_X + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        cy = self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'x': cx, 'y': cy,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(20, 40), 'duration': 40,
                'color': self.np_random.choice([(255, 255, 0), (255, 165, 0), (255, 255, 255)]),
                'size': self.np_random.integers(3, 7)
            })

    def _create_floating_text(self, text, x, y, color=(255,255,255), duration=60, size=1):
        return {
            'text': text, 'x': x, 'y': y, 'color': color, 'life': duration,
            'duration': duration, 'vy': 1.5, 'alpha': 255, 'size': size
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Set up a window to display the game
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Monster Crusher")
    
    # Game loop
    while not terminated:
        # --- Human Input ---
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The obs is already the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # --- Frame Rate ---
        env.clock.tick(30) # Run at 30 FPS

    env.close()