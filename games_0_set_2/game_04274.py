
# Generated: 2025-08-28T01:54:50.864334
# Source Brief: brief_04274.md
# Brief Index: 4274

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game objects
class Enemy:
    def __init__(self, health, speed, path, wave_bonus, np_random):
        self.pos = np.array(path[0], dtype=float)
        self.max_health = health
        self.health = health
        self.speed = speed
        self.path = path
        self.path_index = 0
        self.value = 3 + wave_bonus # Resources given on death
        self.is_alive = True
        self.radius = 6 + int(health / 15)
        self.np_random = np_random

    def move(self):
        if not self.is_alive or self.path_index >= len(self.path) - 1:
            return False  # Reached the end

        target = np.array(self.path[self.path_index + 1], dtype=float)
        direction = target - self.pos
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.pos = target
            self.path_index += 1
        else:
            self.pos += (direction / distance) * self.speed
        
        return self.path_index >= len(self.path) - 1

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.is_alive = False
        return not self.is_alive

class Tower:
    def __init__(self, pos, tower_type_info):
        self.pos = np.array(pos, dtype=float)
        self.stats = tower_type_info
        self.cooldown = 0
        self.target = None

    def update(self, enemies):
        if self.cooldown > 0:
            self.cooldown -= 1
        
        if self.cooldown == 0:
            self.target = self._find_target(enemies)
            if self.target:
                self.cooldown = self.stats['cooldown']
                return self.target, self.stats['damage']
        return None, 0

    def _find_target(self, enemies):
        valid_targets = []
        for enemy in enemies:
            if not enemy.is_alive:
                continue
            distance = np.linalg.norm(self.pos - enemy.pos)
            if distance <= self.stats['range']:
                valid_targets.append(enemy)
        
        if not valid_targets:
            return None
            
        # Target enemy furthest along the path
        return max(valid_targets, key=lambda e: (e.path_index, np.linalg.norm(e.pos - np.array(e.path[e.path_index]))))

class Effect:
    def __init__(self, pos, color, size, lifespan, vel=None, gravity=0.1):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float) if vel is not None else np.array([0.0, 0.0])
        self.color = color
        self.size = size
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.gravity = gravity

    def update(self):
        self.pos += self.vel
        self.vel[1] += self.gravity
        self.lifespan -= 1
        return self.lifespan > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→/↑↓ to select placement spot. Press 'Space' to place tower. Press 'Shift' to cycle tower type."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers along the path."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CRITICAL: Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame and Display ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Visuals ---
        self.FONT_UI = pygame.font.Font(None, 24)
        self.FONT_TITLE = pygame.font.Font(None, 50)
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_PATH = (45, 50, 62)
        self.COLOR_BASE = (60, 179, 113)
        self.COLOR_BASE_GLOW = (90, 200, 140)
        self.COLOR_ENEMY = (255, 70, 85)
        self.COLOR_ENEMY_GLOW = (255, 120, 130)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 255)

        # --- Game Design Constants ---
        self.MAX_STEPS = 3000
        self.TOTAL_WAVES = 10
        self.INITIAL_RESOURCES = 50
        self.INITIAL_BASE_HEALTH = 100
        self.WAVE_PREP_TIME = 150 # frames between waves
        
        self.TOWER_STATS = [
            {'name': 'Gatling', 'color': (60, 160, 255), 'range': 80, 'damage': 5, 'cost': 10, 'cooldown': 10},
            {'name': 'Cannon', 'color': (255, 215, 0), 'range': 120, 'damage': 3, 'cost': 7, 'cooldown': 20},
            {'name': 'Sniper', 'color': (200, 100, 255), 'range': 200, 'damage': 1, 'cost': 5, 'cooldown': 30},
        ]
        
        self.PATH = [(0, 100), (150, 100), (150, 300), (490, 300), (490, 100), (self.WIDTH + 10, 100)]
        self.BASE_POS = (self.WIDTH - 40, 80)
        
        self.TOWER_SPOTS = [
            (100, 150), (200, 150), (200, 250), (300, 250),
            (400, 250), (440, 150), (540, 150), (320, 100), (320, 300)
        ]

        # --- State Variables ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.base_health = 0
        self.resources = 0
        self.current_wave = 0
        self.wave_timer = 0
        self.enemies_in_wave = 0
        self.enemies = []
        self.towers = []
        self.effects = []
        
        # --- UI/Action State ---
        self.cursor_index = 0
        self.selected_tower_type = 0
        self.last_movement = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.current_wave = 0
        self.wave_timer = self.WAVE_PREP_TIME
        self.enemies_in_wave = 0
        
        self.enemies.clear()
        self.towers.clear()
        self.effects.clear()
        
        self.cursor_index = 0
        self.selected_tower_type = 0
        self.last_movement = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            reward += self._handle_input(action)
            
            # --- Game Logic Update ---
            self._update_waves()
            reward += self._update_enemies()
            reward += self._update_towers()
            self._update_effects()
        
        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Cycle Tower Type (on press) ---
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_STATS)
        
        # --- Move Cursor (on new direction) ---
        if movement != 0 and movement != self.last_movement:
            if movement in [1, 3]: # Up or Left
                self.cursor_index = (self.cursor_index - 1 + len(self.TOWER_SPOTS)) % len(self.TOWER_SPOTS)
            elif movement in [2, 4]: # Down or Right
                self.cursor_index = (self.cursor_index + 1) % len(self.TOWER_SPOTS)
        
        # --- Place Tower (on press) ---
        if space_held and not self.last_space_held:
            tower_stat = self.TOWER_STATS[self.selected_tower_type]
            pos = self.TOWER_SPOTS[self.cursor_index]
            
            can_afford = self.resources >= tower_stat['cost']
            is_occupied = any(np.array_equal(t.pos, pos) for t in self.towers)

            if can_afford and not is_occupied:
                self.resources -= tower_stat['cost']
                new_tower = Tower(pos, tower_stat)
                
                # Suboptimal placement check
                is_suboptimal = False
                for t in self.towers:
                    if np.linalg.norm(t.pos - new_tower.pos) < 50:
                        is_suboptimal = True
                        break
                if is_suboptimal:
                    reward -= 0.5
                
                self.towers.append(new_tower)
                # sfx: place_tower.wav
        
        self.last_movement = movement
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        return reward

    def _update_waves(self):
        if self.current_wave == 0 or (len(self.enemies) == 0 and self.enemies_in_wave > 0):
            self.wave_timer -= 1
            if self.wave_timer <= 0 and self.current_wave < self.TOTAL_WAVES:
                self._start_next_wave()

    def _start_next_wave(self):
        self.current_wave += 1
        self.wave_timer = self.WAVE_PREP_TIME
        
        num_enemies = 3 + self.current_wave - 1
        self.enemies_in_wave = num_enemies
        
        for i in range(num_enemies):
            # Speed distribution shifts towards faster enemies
            speed_roll = self.np_random.random()
            speed_threshold = 0.5 - (self.current_wave * 0.1)
            
            if speed_roll < speed_threshold:
                speed, health = 1, 20
            elif speed_roll < 0.85:
                speed, health = 2, 40
            else:
                speed, health = 3, 60
            
            # Stagger enemy spawns
            staggered_path = [tuple(p + np.array([-i * 20, 0])) for p in self.PATH]
            enemy = Enemy(health, speed, staggered_path, self.current_wave, self.np_random)
            self.enemies.append(enemy)

    def _update_enemies(self):
        reward = 0
        for enemy in list(self.enemies):
            if enemy.is_alive:
                reached_end = enemy.move()
                if reached_end:
                    self.base_health = max(0, self.base_health - 10)
                    enemy.is_alive = False
                    # sfx: base_damage.wav
            else:
                self.enemies.remove(enemy)
        return reward

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            target, damage = tower.update(self.enemies)
            if target:
                reward += 0.1 # Reward for hitting
                is_kill = target.take_damage(damage)
                # sfx: projectile_hit.wav
                
                # Projectile visual
                self.effects.append(Effect(tower.pos, (255, 255, 255), 1, 5, vel=(target.pos - tower.pos)/4))

                if is_kill:
                    reward += 1.0 # Reward for kill
                    self.resources += target.value
                    self._create_explosion(target.pos, self.COLOR_ENEMY)
                    # sfx: enemy_explode.wav
        return reward

    def _update_effects(self):
        for effect in list(self.effects):
            if not effect.update():
                self.effects.remove(effect)

    def _create_explosion(self, pos, color):
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 2
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            size = self.np_random.integers(1, 4)
            lifespan = self.np_random.integers(15, 30)
            self.effects.append(Effect(pos, color, size, lifespan, vel, gravity=0.05))

    def _check_termination(self):
        if self.game_over:
            return True
            
        if self.base_health <= 0:
            self.game_over = True
            self.score -= 100
            return True
            
        if self.current_wave >= self.TOTAL_WAVES and len(self.enemies) == 0:
            self.game_over = True
            self.win = True
            self.score += 100
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
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.PATH, 40)
        
        # Base
        base_rect = pygame.Rect(self.BASE_POS[0], self.BASE_POS[1], 40, 40)
        pygame.gfxdraw.box(self.screen, base_rect, (*self.COLOR_BASE, 150))
        pygame.gfxdraw.rectangle(self.screen, base_rect, self.COLOR_BASE_GLOW)
        
        # Tower spots
        for i, pos in enumerate(self.TOWER_SPOTS):
            is_occupied = any(np.array_equal(t.pos, pos) for t in self.towers)
            color = (*self.TOWER_STATS[self.selected_tower_type]['color'], 20) if is_occupied else (100, 100, 100, 30)
            if i == self.cursor_index and not is_occupied:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 15, color)
            else:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 15, color)

        # Towers
        for tower in self.towers:
            pos_int = (int(tower.pos[0]), int(tower.pos[1]))
            color = tower.stats['color']
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 10, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 10, color)
            # Range indicator when selected
            if np.array_equal(tower.pos, self.TOWER_SPOTS[self.cursor_index]):
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], tower.stats['range'], (*color, 60))

        # Enemies
        for enemy in self.enemies:
            pos_int = (int(enemy.pos[0]), int(enemy.pos[1]))
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], enemy.radius, (*self.COLOR_ENEMY_GLOW, 50))
            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], enemy.radius - 2, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], enemy.radius - 2, self.COLOR_ENEMY_GLOW)

        # Effects
        for effect in self.effects:
            pos_int = (int(effect.pos[0]), int(effect.pos[1]))
            alpha = int(255 * (effect.lifespan / effect.max_lifespan))
            color = (*effect.color, alpha)
            if effect.size > 1:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], effect.size, color)
            else:
                pygame.gfxdraw.pixel(self.screen, pos_int[0], pos_int[1], color)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.base_health / self.INITIAL_BASE_HEALTH
        bar_width = 200
        health_width = int(bar_width * health_ratio)
        pygame.draw.rect(self.screen, (80, 0, 0), (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, (0, 180, 0), (10, 10, health_width, 20))
        health_text = self.FONT_UI.render(f'BASE HP: {self.base_health}', True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))
        
        # Resources
        res_text = self.FONT_UI.render(f'CREDITS: {self.resources}', True, self.COLOR_TEXT)
        self.screen.blit(res_text, (10, self.HEIGHT - 30))
        
        # Wave Info
        wave_str = f'WAVE: {self.current_wave}/{self.TOTAL_WAVES}' if self.current_wave > 0 else "GET READY"
        wave_text = self.FONT_UI.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Cursor
        cursor_pos = self.TOWER_SPOTS[self.cursor_index]
        pygame.gfxdraw.aacircle(self.screen, int(cursor_pos[0]), int(cursor_pos[1]), 18, self.COLOR_CURSOR)

        # Selected Tower UI
        stat = self.TOWER_STATS[self.selected_tower_type]
        pygame.draw.rect(self.screen, (40,40,50), (10, self.HEIGHT - 100, 200, 60))
        pygame.draw.rect(self.screen, (60,60,70), (10, self.HEIGHT - 100, 200, 60), 2)
        
        type_text = self.FONT_UI.render(f"Tower: {stat['name']}", True, stat['color'])
        cost_text = self.FONT_UI.render(f"Cost: {stat['cost']}", True, self.COLOR_TEXT)
        self.screen.blit(type_text, (20, self.HEIGHT - 95))
        self.screen.blit(cost_text, (20, self.HEIGHT - 75))

        # Game Over / Win Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.FONT_TITLE.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "resources": self.resources,
            "base_health": self.base_health,
        }

    def close(self):
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
        
        # Test specific mechanics
        self.reset()
        assert self.base_health == self.INITIAL_BASE_HEALTH, "Base health not reset"
        assert self.resources == self.INITIAL_RESOURCES, "Resources not reset"
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    # For headless execution
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually (requires a display) ---
    # To run this part, comment out the "dummy" SDL_VIDEODRIVER line above
    # and uncomment the following block.
    
    # os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    # pygame.display.set_caption("Tower Defense")
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # obs, info = env.reset()
    # done = False
    # while not done:
    #     movement, space, shift = 0, 0, 0
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_UP: movement = 1
    #             if event.key == pygame.K_DOWN: movement = 2
    #             if event.key == pygame.K_LEFT: movement = 3
    #             if event.key == pygame.K_RIGHT: movement = 4
    #             if event.key == pygame.K_SPACE: space = 1
    #             if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
    
    #     action = [movement, space, shift]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated

    #     # Draw the observation to the display
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #     env.clock.tick(30) # Limit to 30 FPS
    
    # env.close()