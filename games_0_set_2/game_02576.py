
# Generated: 2025-08-27T20:47:00.339000
# Source Brief: brief_02576.md
# Brief Index: 2576

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move cursor. Space to place tower. Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from zombie waves by strategically placing towers. Survive 5 rounds to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 40
    GRID_W = SCREEN_WIDTH // GRID_SIZE
    GRID_H = SCREEN_HEIGHT // GRID_SIZE
    MAX_STEPS = 30 * 120 # 2 minutes at 30fps
    NUM_ROUNDS_TO_WIN = 5

    # Colors
    COLOR_BG = (25, 28, 32)
    COLOR_GRID = (40, 44, 52)
    COLOR_BASE = (0, 80, 120)
    COLOR_PLAYER_CURSOR = (100, 200, 255)
    COLOR_PLAYER_CURSOR_INVALID = (255, 100, 100)
    COLOR_ZOMBIE = (220, 50, 50)
    COLOR_ZOMBIE_HEALTH_BG = (50, 50, 50)
    COLOR_ZOMBIE_HEALTH = (50, 220, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_TOWER_GUN = (180, 180, 190)
    COLOR_TOWER_CANNON = (150, 150, 160)
    COLOR_PROJECTILE_GUN = (255, 255, 0)
    COLOR_PROJECTILE_CANNON = (255, 180, 0)

    TOWER_SPECS = {
        0: {"name": "Gatling", "cost": 25, "range": 80, "damage": 5, "fire_rate": 5, "unlock_round": 1},
        1: {"name": "Cannon", "cost": 75, "range": 120, "damage": 40, "fire_rate": 0.8, "unlock_round": 2},
        2: {"name": "Frost", "cost": 50, "range": 100, "damage": 1, "fire_rate": 2, "unlock_round": 3, "slow": 0.5}
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.render_mode = render_mode
        self.np_random = None

        # These attributes are defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.resources = 0
        self.current_round = 0
        self.zombies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type_idx = 0
        self.available_tower_types = []
        self.zombie_spawn_timer = 0
        self.zombies_to_spawn_in_wave = 0
        self.zombies_killed_in_wave = 0
        self.wave_zombie_base_count = 0
        self.game_won = False
        self.last_action = [0, 0, 0]
        self.round_end_timer = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = 100
        self.resources = 80
        
        self.zombies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2 -1]
        self.selected_tower_type_idx = 0
        self.last_action = [0, 0, 0]
        
        self.current_round = 0
        self._start_new_round()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_press, shift_press = action[0], action[1], action[2]
        reward = -0.001 # Small penalty for existing

        if not self.game_over:
            self._handle_input(movement, space_press, shift_press)
            
            self._update_round_timer()
            self._spawn_zombies()
            self._update_towers()
            self._update_projectiles()
            reward += self._update_zombies()
            self._update_particles()
            
            reward += self._check_zombies_at_base()
            
            if self._check_round_end():
                reward += 5 # Round survived bonus
                self.round_end_timer = 3 * 30 # 3 second delay
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_won:
            reward -= 50 # Loss penalty
        elif self.game_won:
            reward += 50 # Win bonus
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_press, shift_press):
        # Movement is continuous
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 2) # Can't build on last row

        # Actions are on press (state change from 0 to 1)
        if space_press and not self.last_action[1]:
            self._place_tower()
        if shift_press and not self.last_action[2]:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.available_tower_types)
            # sfx: UI_Cycle.wav

        self.last_action = [movement, space_press, shift_press]
        
    def _start_new_round(self):
        self.current_round += 1
        if self.current_round > self.NUM_ROUNDS_TO_WIN:
            self.game_won = True
            return

        self.wave_zombie_base_count = int(8 * (1.2 ** (self.current_round - 1)))
        self.zombies_to_spawn_in_wave = self.wave_zombie_base_count
        self.zombies_killed_in_wave = 0
        self.zombie_spawn_timer = 1 * 30 # 1 second initial delay
        
        # Unlock new towers
        self.available_tower_types = [tid for tid, spec in self.TOWER_SPECS.items() if self.current_round >= spec["unlock_round"]]
        self.selected_tower_type_idx = min(self.selected_tower_type_idx, len(self.available_tower_types) - 1)

    def _update_round_timer(self):
        if self.round_end_timer > 0:
            self.round_end_timer -= 1
            if self.round_end_timer == 0:
                self._start_new_round()
    
    def _check_round_end(self):
        return self.zombies_to_spawn_in_wave == 0 and not self.zombies and self.round_end_timer == 0 and not self.game_won

    def _spawn_zombies(self):
        if self.zombies_to_spawn_in_wave > 0 and self.round_end_timer == 0:
            self.zombie_spawn_timer -= 1
            if self.zombie_spawn_timer <= 0:
                spawn_x = self.np_random.integers(0, self.SCREEN_WIDTH)
                speed = 0.5 + self.current_round * 0.1
                health = 20 * (1.2 ** (self.current_round - 1))
                
                self.zombies.append({
                    "pos": np.array([spawn_x, -10.0]), "health": health, "max_health": health,
                    "speed": speed, "slow_timer": 0, "id": self.np_random.random()
                })
                self.zombies_to_spawn_in_wave -= 1
                self.zombie_spawn_timer = max(10, 30 - self.current_round * 2) # Spawn faster in later rounds

    def _place_tower(self):
        tower_type_id = self.available_tower_types[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[tower_type_id]
        cost = spec["cost"]
        
        is_occupied = any(t['grid_pos'] == self.cursor_pos for t in self.towers)

        if self.resources >= cost and not is_occupied:
            self.resources -= cost
            self.towers.append({
                "grid_pos": list(self.cursor_pos), "type": tower_type_id,
                "range_sq": spec["range"] ** 2, "cooldown": 0,
                "target_id": None
            })
            # sfx: Tower_Place.wav
    
    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            spec = self.TOWER_SPECS[tower['type']]
            center_pos = np.array(tower['grid_pos']) * self.GRID_SIZE + self.GRID_SIZE / 2

            if tower['cooldown'] <= 0:
                target = None
                min_dist_sq = float('inf')
                
                for z in self.zombies:
                    dist_sq = np.sum((z['pos'] - center_pos)**2)
                    if dist_sq < tower['range_sq'] and dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        target = z
                
                if target:
                    tower['cooldown'] = 30 / spec['fire_rate']
                    tower['target_id'] = target['id']
                    self.projectiles.append({
                        "pos": center_pos.copy(), "target_id": target['id'], "type": tower['type'], "speed": 10
                    })
                    # sfx: Gun_Fire.wav or Cannon_Fire.wav

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            target_zombie = next((z for z in self.zombies if z['id'] == p['target_id']), None)
            
            if not target_zombie:
                self.projectiles.remove(p)
                continue
            
            direction = target_zombie['pos'] - p['pos']
            dist = np.linalg.norm(direction)
            
            if dist < p['speed']:
                self._on_projectile_hit(p, target_zombie)
                self.projectiles.remove(p)
            else:
                p['pos'] += (direction / dist) * p['speed']
    
    def _on_projectile_hit(self, projectile, zombie):
        spec = self.TOWER_SPECS[projectile['type']]
        zombie['health'] -= spec['damage']
        # sfx: Zombie_Hit.wav
        
        if 'slow' in spec:
            zombie['slow_timer'] = max(zombie['slow_timer'], 2 * 30) # 2 second slow

        # Create hit particle effect
        for _ in range(5):
            self.particles.append(self._create_particle(zombie['pos'], self.COLOR_PROJECTILE_GUN if projectile['type'] == 0 else self.COLOR_PROJECTILE_CANNON))

    def _update_zombies(self):
        kill_reward = 0
        for z in self.zombies[:]:
            current_speed = z['speed']
            if z['slow_timer'] > 0:
                z['slow_timer'] -= 1
                slow_effect = self.TOWER_SPECS[2]['slow']
                current_speed *= (1 - slow_effect)

            z['pos'][1] += current_speed
            
            if z['health'] <= 0:
                self.zombies.remove(z)
                self.zombies_killed_in_wave += 1
                kill_reward += 1.0
                self.resources += 5
                # sfx: Zombie_Die.wav
                # Create death particle effect
                for _ in range(20):
                    self.particles.append(self._create_particle(z['pos'], self.COLOR_ZOMBIE, 15))
        return kill_reward

    def _check_zombies_at_base(self):
        damage_reward = 0
        for z in self.zombies[:]:
            if z['pos'][1] > self.SCREEN_HEIGHT - self.GRID_SIZE:
                self.zombies.remove(z)
                self.base_health -= 10
                damage_reward -= 5
                # sfx: Base_Damage.wav
                # Create base hit particle effect
                base_hit_pos = np.array([z['pos'][0], self.SCREEN_HEIGHT - self.GRID_SIZE])
                for _ in range(30):
                    self.particles.append(self._create_particle(base_hit_pos, self.COLOR_BASE, 20))
        return damage_reward

    def _create_particle(self, pos, color, initial_lifespan=10):
        return {
            "pos": pos.copy(),
            "vel": self.np_random.uniform(-1, 1, 2) * 2,
            "life": initial_lifespan + self.np_random.integers(0, 5),
            "color": color
        }

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.9
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.game_over: return True
        if self.base_health <= 0 or self.steps >= self.MAX_STEPS or self.game_won:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "round": self.current_round,
            "base_health": self.base_health,
            "resources": self.resources,
            "zombies_left": len(self.zombies) + self.zombies_to_spawn_in_wave
        }
    
    def _render_game(self):
        # Grid
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Base
        base_rect = pygame.Rect(0, self.SCREEN_HEIGHT - self.GRID_SIZE, self.SCREEN_WIDTH, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

        # Towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            pos = (int(tower['grid_pos'][0] * self.GRID_SIZE + self.GRID_SIZE / 2), 
                   int(tower['grid_pos'][1] * self.GRID_SIZE + self.GRID_SIZE / 2))
            color = self.COLOR_TOWER_GUN if spec['name'] == "Gatling" else self.COLOR_TOWER_CANNON
            if spec['name'] == "Frost": color = (100, 150, 255)
            
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.GRID_SIZE // 2 - 4, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.GRID_SIZE // 2 - 4, color)

        # Projectiles
        for p in self.projectiles:
            color = self.COLOR_PROJECTILE_GUN if p['type'] == 0 else self.COLOR_PROJECTILE_CANNON
            if p['type'] == 2: color = (180, 220, 255)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, color)
            
        # Zombies
        for z in self.zombies:
            pos = (int(z['pos'][0]), int(z['pos'][1]))
            radius = 8
            color = self.COLOR_ZOMBIE
            if z['slow_timer'] > 0:
                color = (150, 50, 220) # Purple when slowed
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            
            # Health bar
            health_pct = z['health'] / z['max_health']
            bar_w = 20
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE_HEALTH_BG, (pos[0] - bar_w/2, pos[1] - 15, bar_w, 4))
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE_HEALTH, (pos[0] - bar_w/2, pos[1] - 15, bar_w * health_pct, 4))

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 10))))
            color_with_alpha = p['color'] + (alpha,)
            size = int(p['life'] / 2)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color_with_alpha)

        # Cursor
        self._render_cursor()

    def _render_cursor(self):
        cursor_px_pos = (self.cursor_pos[0] * self.GRID_SIZE, self.cursor_pos[1] * self.GRID_SIZE)
        tower_type_id = self.available_tower_types[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[tower_type_id]
        cost = spec['cost']
        is_occupied = any(t['grid_pos'] == self.cursor_pos for t in self.towers)
        can_afford = self.resources >= cost
        is_valid = can_afford and not is_occupied

        color = self.COLOR_PLAYER_CURSOR if is_valid else self.COLOR_PLAYER_CURSOR_INVALID
        
        # Draw placement box
        rect = pygame.Rect(cursor_px_pos, (self.GRID_SIZE, self.GRID_SIZE))
        pygame.draw.rect(self.screen, color, rect, 2)
        
        # Draw range indicator
        center_px = (cursor_px_pos[0] + self.GRID_SIZE//2, cursor_px_pos[1] + self.GRID_SIZE//2)
        pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], spec['range'], color + (100,))

    def _render_ui(self):
        # Top-left info panel
        info_texts = [
            f"HEALTH: {max(0, self.base_health)}%",
            f"ROUND: {self.current_round}/{self.NUM_ROUNDS_TO_WIN}",
            f"RESOURCES: ${self.resources}",
            f"SCORE: {int(self.score)}"
        ]
        for i, text in enumerate(info_texts):
            surf = self.font_small.render(text, True, self.COLOR_UI_TEXT)
            self.screen.blit(surf, (10, 10 + i * 20))

        # Selected Tower Info (near cursor)
        tower_type_id = self.available_tower_types[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[tower_type_id]
        cost = spec['cost']
        is_occupied = any(t['grid_pos'] == self.cursor_pos for t in self.towers)
        can_afford = self.resources >= cost
        
        tower_text = f"[{spec['name']}] Cost: ${cost}"
        color = self.COLOR_UI_TEXT
        if not can_afford: color = self.COLOR_PLAYER_CURSOR_INVALID
        if is_occupied: tower_text = "[OCCUPIED]"

        surf = self.font_small.render(tower_text, True, color)
        text_pos_x = self.cursor_pos[0] * self.GRID_SIZE
        text_pos_y = self.cursor_pos[1] * self.GRID_SIZE - 20
        text_pos_x = np.clip(text_pos_x, 0, self.SCREEN_WIDTH - surf.get_width())
        text_pos_y = np.clip(text_pos_y, 0, self.SCREEN_HEIGHT - surf.get_height())
        self.screen.blit(surf, (text_pos_x, text_pos_y))

        # Round transition text
        if self.round_end_timer > 0:
            text = f"ROUND {self.current_round} CLEARED!" if self.current_round <= self.NUM_ROUNDS_TO_WIN else ""
            if self.current_round < self.NUM_ROUNDS_TO_WIN:
                text2 = f"Next round in {math.ceil(self.round_end_timer/30)}..."
            else:
                text2 = ""
            
            surf = self.font_large.render(text, True, self.COLOR_UI_TEXT)
            self.screen.blit(surf, (self.SCREEN_WIDTH/2 - surf.get_width()/2, self.SCREEN_HEIGHT/2 - 30))
            surf2 = self.font_small.render(text2, True, self.COLOR_UI_TEXT)
            self.screen.blit(surf2, (self.SCREEN_WIDTH/2 - surf2.get_width()/2, self.SCREEN_HEIGHT/2))

        # Game Over / Win Text
        if self.game_over:
            text = "YOU WIN!" if self.game_won else "GAME OVER"
            surf = self.font_large.render(text, True, self.COLOR_UI_TEXT)
            self.screen.blit(surf, (self.SCREEN_WIDTH/2 - surf.get_width()/2, self.SCREEN_HEIGHT/2 - 20))


    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    # For manual play
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop for human play
    while not terminated:
        # Action mapping for keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render to screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        env.clock.tick(30)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()