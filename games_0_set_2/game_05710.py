
# Generated: 2025-08-28T05:52:42.338760
# Source Brief: brief_05710.md
# Brief Index: 5710

        
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
        "Controls: ↑↓←→ to move. Press space to attack adjacent squares. "
        "Shift does nothing."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defeat waves of monsters in a grid-based arena, collecting coins and maximizing your score through risky plays."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 400, 400
        self.GRID_SIZE = 20
        self.CELL_SIZE = self.GRID_WIDTH // self.GRID_SIZE
        self.MAX_STEPS = 1000
        self.INITIAL_PLAYER_HEALTH = 10

        # --- Colors ---
        self.COLOR_BG = (15, 25, 40) # Dark Blue
        self.COLOR_GRID = (30, 50, 80) # Lighter Blue
        self.COLOR_UI_BG = (25, 35, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_PLAYER = (50, 205, 50) # Lime Green
        self.COLOR_PLAYER_ATTACK = (255, 255, 100)
        self.COLOR_COIN = (255, 223, 0) # Gold
        self.COLOR_HEALTH_FG = (50, 205, 50)
        self.COLOR_HEALTH_BG = (205, 50, 50)
        self.MONSTER_COLORS = {
            "basic": (255, 70, 70), # Red
            "speedy": (180, 80, 255), # Purple
            "tank": (255, 165, 0), # Orange
        }

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 40)
        self.font_title = pygame.font.Font(None, 32)
        
        # Etc...        
        self.player_pos = None
        self.player_health = None
        self.monsters = None
        self.coins = None
        self.particles = None
        self.score = None
        self.wave = None
        self.steps = None
        self.game_over = None
        self.last_action_info = None
        
        # Initialize state variables
        self.reset()

        # Run self-check after full initialization
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.player_health = self.INITIAL_PLAYER_HEALTH
        self.monsters = []
        self.coins = []
        self.particles = []
        self.score = 0
        self.wave = 0 # Will be incremented to 1 by _spawn_wave
        self.steps = 0
        self.game_over = False
        self.last_action_info = "Game Start"

        self._spawn_wave()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Update Particles ---
        self._update_particles()
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # --- Player Turn (Priority: Attack > Move > Wait) ---
        action_taken = False
        if space_held:
            self.last_action_info = "Attack!"
            action_taken = True
            attacked_cells = self._get_adjacent_cells(self.player_pos, include_diagonals=True)
            
            # Sound placeholder: # pygame.mixer.Sound.play(self.attack_sound)
            self._create_attack_particles(attacked_cells)

            monsters_to_remove = []
            for monster in self.monsters:
                if monster['pos'] in attacked_cells:
                    monster['health'] -= 1
                    if monster['health'] <= 0:
                        monsters_to_remove.append(monster)
                        # Sound placeholder: # pygame.mixer.Sound.play(self.monster_die_sound)
                        self._create_death_particles(monster['pos'], self.MONSTER_COLORS[monster['type']])
                        
                        if monster['type'] == 'basic': reward += 1; self.score += 10
                        elif monster['type'] == 'speedy': reward += 2; self.score += 20
                        elif monster['type'] == 'tank': reward += 5; self.score += 50
            self.monsters = [m for m in self.monsters if m not in monsters_to_remove]

        elif movement != 0:
            action_taken = True
            dx, dy = 0, 0
            if movement == 1: dy, name = -1, "Move Up"
            elif movement == 2: dy, name = 1, "Move Down"
            elif movement == 3: dx, name = -1, "Move Left"
            elif movement == 4: dx, name = 1, "Move Right"
            
            self.last_action_info = name
            self.player_pos[0] = np.clip(self.player_pos[0] + dx, 0, self.GRID_SIZE - 1)
            self.player_pos[1] = np.clip(self.player_pos[1] + dy, 0, self.GRID_SIZE - 1)

            if any(self._is_adjacent(self.player_pos, m['pos']) for m in self.monsters):
                reward -= 0.1

        if not action_taken:
            self.last_action_info = "Wait"; reward -= 0.2

        # --- Monster Turn & Collisions ---
        self._update_monsters()
        
        damage_taken = sum(1 for m in self.monsters if m['pos'] == self.player_pos)
        if damage_taken > 0:
            self.player_health -= damage_taken
            # Sound placeholder: # pygame.mixer.Sound.play(self.player_hurt_sound)
            self._create_damage_particles(self.player_pos)

        coins_collected = sum(1 for c in self.coins if c == self.player_pos)
        if coins_collected > 0:
            self.coins = [c for c in self.coins if c != self.player_pos]
            reward += 0.1 * coins_collected
            self.score += 1 * coins_collected
            # Sound placeholder: # pygame.mixer.Sound.play(self.coin_collect_sound)

        # --- Wave & Termination Check ---
        if not self.monsters:
            reward += 100; self.score += 50 * self.wave
            # Sound placeholder: # pygame.mixer.Sound.play(self.wave_clear_sound)
            self._spawn_wave()
            self.last_action_info = f"Wave {self.wave} Start!"
        
        terminated = False
        if self.player_health <= 0:
            self.player_health = 0; reward -= 100; terminated = True
            self.game_over = True; self.last_action_info = "Game Over"
        if self.steps >= self.MAX_STEPS:
            terminated = True; self.game_over = True
            self.last_action_info = "Time Limit Reached"
        
        # MUST return exactly this 5-tuple
        return (self._get_observation(), reward, terminated, False, self._get_info())
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Grid
        for i in range(self.GRID_SIZE + 1):
            x = i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.GRID_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, x), (self.GRID_WIDTH, x))
            
        # Coins
        pulse = abs(math.sin(self.steps * 0.2)) * 3
        for x, y in self.coins:
            cx, cy = int((x + 0.5) * self.CELL_SIZE), int((y + 0.5) * self.CELL_SIZE)
            r = int(self.CELL_SIZE * 0.3 + pulse)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, r, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, r, self.COLOR_COIN)

        # Monsters
        for m in self.monsters:
            rect = pygame.Rect(m['pos'][0] * self.CELL_SIZE + 2, m['pos'][1] * self.CELL_SIZE + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
            pygame.draw.rect(self.screen, self.MONSTER_COLORS[m['type']], rect, border_radius=3)
            if m['type'] == 'tank': pygame.draw.rect(self.screen, (255,255,255), rect.inflate(-6, -6), 2, border_radius=2)

        # Player
        px, py = self.player_pos
        player_rect = pygame.Rect(px * self.CELL_SIZE, py * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-2,-2), border_radius=3)
        pygame.draw.circle(self.screen, (255, 255, 255), (int((px + 0.5) * self.CELL_SIZE), int((py + 0.3) * self.CELL_SIZE)), 2)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['size']))

    def _render_ui(self):
        # UI Panel
        ui_rect = pygame.Rect(self.GRID_WIDTH, 0, self.WIDTH - self.GRID_WIDTH, self.HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_WIDTH, 0), (self.GRID_WIDTH, self.HEIGHT), 2)

        # UI Text
        self.screen.blit(self.font_title.render("GRID SLAYER", True, self.COLOR_TEXT), (self.GRID_WIDTH + 20, 20))
        self.screen.blit(self.font_large.render(f"{self.score:06d}", True, self.COLOR_COIN), (self.GRID_WIDTH + 20, 70))
        self.screen.blit(self.font_small.render("WAVE", True, self.COLOR_TEXT), (self.GRID_WIDTH + 20, 120))
        self.screen.blit(self.font_large.render(f"{self.wave}", True, self.COLOR_TEXT), (self.GRID_WIDTH + 20, 140))
        self.screen.blit(self.font_small.render("HEALTH", True, self.COLOR_TEXT), (self.GRID_WIDTH + 20, 200))

        # Health Bar
        health_ratio = max(0, self.player_health / self.INITIAL_PLAYER_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (self.GRID_WIDTH + 20, 230, 200, 25), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (self.GRID_WIDTH + 20, 230, int(200 * health_ratio), 25), border_radius=5)
        
        self.screen.blit(self.font_small.render("Last Action:", True, self.COLOR_TEXT), (self.GRID_WIDTH + 20, 300))
        self.screen.blit(self.font_small.render(self.last_action_info, True, (200, 200, 100)), (self.GRID_WIDTH + 20, 325))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.player_health,
        }

    def _spawn_wave(self):
        self.wave += 1
        num_monsters = 2 + self.wave
        
        self.coins.clear()
        for _ in range(self.np_random.integers(3, 6)):
            self.coins.append(self._get_random_empty_cell())

        for _ in range(num_monsters):
            monster_type = "basic"; health = 1
            if self.wave >= 3 and self.np_random.random() < 0.1: monster_type = "speedy"
            if self.wave >= 5 and self.np_random.random() < 0.05: monster_type = "tank"; health = 3
            
            self.monsters.append({
                'pos': self._get_random_empty_cell(), 'type': monster_type, 'health': health,
                'dir': (self.np_random.choice([-1, 0, 1]), self.np_random.choice([-1, 0, 1])) if monster_type == 'speedy' else (0,0)
            })

    def _update_monsters(self):
        for m in self.monsters:
            if m['type'] in ['basic', 'tank']:
                dx, dy = self.np_random.integers(-1, 2), self.np_random.integers(-1, 2)
                m['pos'][0] = np.clip(m['pos'][0] + dx, 0, self.GRID_SIZE - 1)
                m['pos'][1] = np.clip(m['pos'][1] + dy, 0, self.GRID_SIZE - 1)
            elif m['type'] == 'speedy':
                dx, dy = m['dir']
                if dx == 0 and dy == 0: dx, dy = self.np_random.choice([-1, 0, 1]), self.np_random.choice([-1, 0, 1]); m['dir'] = (dx, dy)
                
                new_pos = [np.clip(m['pos'][0] + dx*2, 0, self.GRID_SIZE - 1), np.clip(m['pos'][1] + dy*2, 0, self.GRID_SIZE - 1)]
                if m['pos'] == new_pos: m['dir'] = (self.np_random.choice([-1, 0, 1]), self.np_random.choice([-1, 0, 1]))
                m['pos'] = new_pos

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['ttl'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]; p['pos'][1] += p['vel'][1]
            p['ttl'] -= 1; p['size'] = max(0, p['size'] - 0.1)

    def _create_particles(self, pos, color, count, speed_range, ttl_range, size_range):
        px, py = (pos[0] + 0.5) * self.CELL_SIZE, (pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(*speed_range)
            self.particles.append({'pos': [px, py], 'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'ttl': self.np_random.integers(*ttl_range), 'size': self.np_random.uniform(*size_range), 'color': color})

    def _create_attack_particles(self, cells):
        for cell in cells: self._create_particles(cell, self.COLOR_PLAYER_ATTACK, 5, (1,3), (5,10), (2,4))
    def _create_death_particles(self, pos, color): self._create_particles(pos, color, 30, (1,5), (10,20), (2,5))
    def _create_damage_particles(self, pos): self._create_particles(pos, self.COLOR_HEALTH_BG, 15, (1,4), (8,15), (1,4))

    def _get_adjacent_cells(self, pos, include_diagonals=False):
        cells = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0: continue
                if not include_diagonals and abs(dx) + abs(dy) > 1: continue
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE: cells.append([nx, ny])
        return cells

    def _is_adjacent(self, pos1, pos2): return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) == 1

    def _get_random_empty_cell(self):
        occupied = {tuple(m['pos']) for m in self.monsters} | {tuple(self.player_pos)} | {tuple(c) for c in self.coins}
        while True:
            pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
            if pos not in occupied: return list(pos)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
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
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(env.game_description)
    
    terminated = False
    while not terminated:
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        movement, space, shift = 0, 0, 0
        
        event = pygame.event.wait() 
        if event.type == pygame.QUIT:
            terminated = True
        elif event.type == pygame.KEYDOWN:
            key_map = {pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4}
            movement = key_map.get(event.key, 0)
            if event.key == pygame.K_SPACE: space = 1
            
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")
            if terminated:
                print("Game Over!")
                # Show final screen for 2 seconds
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                pygame.time.wait(2000)

    env.close()