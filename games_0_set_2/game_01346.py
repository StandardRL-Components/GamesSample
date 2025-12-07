
# Generated: 2025-08-27T16:50:52.366365
# Source Brief: brief_01346.md
# Brief Index: 1346

        
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
        "Controls: Use arrow keys to select a build location. Press Shift to cycle tower types. Press Space to build."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing defensive towers along their path."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 30 * 180 # 3 minutes max
        self.MAX_WAVES = 10

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PATH = (40, 45, 50)
        self.COLOR_PATH_BORDER = (60, 65, 70)
        self.COLOR_BASE = (0, 150, 200)
        self.COLOR_BASE_GLOW = (0, 150, 200, 50)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_ENEMY_GLOW = (220, 50, 50, 100)
        self.COLOR_TOWER_SPOT = (0, 100, 50, 100)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (40, 200, 40)
        self.COLOR_HEALTH_BAR_BG = (100, 40, 40)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Game entities
        self.path = []
        self.placement_spots = []
        self.tower_types = []
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        # State variables are initialized in reset()
        self.reset()
        
        # Used to detect key presses vs holds
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.validate_implementation()

    def _define_level(self):
        self.path = [
            (-20, 100), (100, 100), (100, 300), (300, 300), 
            (300, 50), (500, 50), (500, 250), (self.WIDTH + 20, 250)
        ]
        self.base_pos = (self.WIDTH - 40, 250)
        self.placement_spots = [
            (50, 150), (150, 100), (150, 250), (250, 250), 
            (350, 100), (450, 100), (450, 200)
        ]
        self.tower_types = [
            {
                "name": "Cannon", "cost": 100, "range": 80, "damage": 25, 
                "fire_rate": 45, "color": (50, 100, 255), "proj_speed": 5
            },
            {
                "name": "Gatling", "cost": 150, "range": 60, "damage": 10, 
                "fire_rate": 10, "color": (255, 150, 0), "proj_speed": 7
            }
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._define_level()

        self.steps = 0
        self.score = 0
        self.money = 200
        self.base_health = 100
        self.max_base_health = 100
        self.game_over = False
        self.win = False

        self.wave_number = 0
        self.wave_in_progress = False
        self.enemies_to_spawn = []
        self.spawn_timer = 0

        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.cursor_pos_index = 0
        self.selected_tower_type = 0

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for each time step
        
        self._handle_input(action)
        
        reward += self._update_game_state()

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                reward += 100.0
            else:
                reward -= 50.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement: Select placement spot ---
        if movement in [1, 3]: # Up or Left
            self.cursor_pos_index = (self.cursor_pos_index - 1) % len(self.placement_spots)
        elif movement in [2, 4]: # Down or Right
            self.cursor_pos_index = (self.cursor_pos_index + 1) % len(self.placement_spots)

        # --- Space: Place tower (on key press) ---
        if space_held and not self.prev_space_held:
            spot_pos = self.placement_spots[self.cursor_pos_index]
            tower_type = self.tower_types[self.selected_tower_type]
            
            is_occupied = any(t['pos'] == spot_pos for t in self.towers)
            
            if not is_occupied and self.money >= tower_type['cost']:
                self.money -= tower_type['cost']
                self.towers.append({
                    "pos": spot_pos,
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                })
                # sfx: build_tower

        # --- Shift: Cycle tower type (on key press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_types)
            # sfx: cycle_weapon

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_game_state(self):
        step_reward = 0.0
        
        # Spawn enemies
        if self.wave_in_progress:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0 and self.enemies_to_spawn:
                enemy_data = self.enemies_to_spawn.pop(0)
                self.enemies.append({
                    "pos": np.array(self.path[0], dtype=float),
                    "health": enemy_data['health'],
                    "max_health": enemy_data['health'],
                    "speed": enemy_data['speed'],
                    "path_index": 1,
                })
                self.spawn_timer = 30 # Spawn every second

        # Update towers
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                spec = self.tower_types[tower['type']]
                target = self._find_target(tower['pos'], spec['range'])
                if target:
                    self.projectiles.append({
                        "pos": np.array(tower['pos'], dtype=float),
                        "target": target,
                        "speed": spec['proj_speed'],
                        "damage": spec['damage'],
                        "color": spec['color']
                    })
                    tower['cooldown'] = spec['fire_rate']
                    # sfx: tower_shoot

        # Update projectiles
        for proj in self.projectiles[:]:
            if proj['target']['health'] <= 0: # Target already dead
                self.projectiles.remove(proj)
                continue
            
            direction = proj['target']['pos'] - proj['pos']
            dist = np.linalg.norm(direction)
            if dist < proj['speed']:
                proj['target']['health'] -= proj['damage']
                step_reward += 0.1 # Reward for hitting
                self._create_particles(proj['pos'], proj['color'], 5)
                self.projectiles.remove(proj)
                # sfx: projectile_hit
            else:
                proj['pos'] += (direction / dist) * proj['speed']

        # Update enemies
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                self.score += 10
                self.money += 15
                step_reward += 1.0 # Reward for kill
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 15)
                self.enemies.remove(enemy)
                # sfx: enemy_death
                continue

            target_pos = self.path[enemy['path_index']]
            direction = np.array(target_pos) - enemy['pos']
            dist = np.linalg.norm(direction)

            if dist < enemy['speed']:
                enemy['pos'] = np.array(target_pos, dtype=float)
                enemy['path_index'] += 1
                if enemy['path_index'] >= len(self.path):
                    self.base_health -= 10
                    self.enemies.remove(enemy)
                    self._create_particles(self.base_pos, self.COLOR_BASE, 20, 1.5)
                    # sfx: base_damage
            else:
                enemy['pos'] += (direction / dist) * enemy['speed']

        # Update particles
        for p in self.particles[:]:
            p['life'] -= 1
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            if p['life'] <= 0:
                self.particles.remove(p)

        # Check for wave completion
        if self.wave_in_progress and not self.enemies and not self.enemies_to_spawn:
            self.wave_in_progress = False
            self.score += 100
            step_reward += 50.0 # Reward for clearing a wave
            if self.wave_number < self.MAX_WAVES:
                self._start_next_wave()
            else:
                self.win = True
        
        return step_reward

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return
        
        num_enemies = 5 + (self.wave_number - 1) * 2
        enemy_speed = 1.0 + (self.wave_number - 1) * 0.05
        enemy_health = 50 + (self.wave_number - 1) * 15
        
        self.enemies_to_spawn = [{
            'health': enemy_health, 'speed': enemy_speed
        } for _ in range(num_enemies)]
        
        self.spawn_timer = 90 # 3 second delay before wave starts
        self.wave_in_progress = True

    def _find_target(self, tower_pos, tower_range):
        for enemy in self.enemies:
            if np.linalg.norm(enemy['pos'] - tower_pos) <= tower_range:
                return enemy
        return None

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                "pos": np.array(pos, dtype=float),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                "life": self.np_random.integers(10, 20),
                "color": color,
                "radius": self.np_random.uniform(1, 3)
            })

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.win:
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

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "money": self.money,
            "win": self.win,
        }
        
    def _render_game(self):
        # Draw path
        for i in range(len(self.path) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, self.path[i], self.path[i+1], 44)
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path[i], self.path[i+1], 40)

        # Draw placement spots
        for pos in self.placement_spots:
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 15, self.COLOR_TOWER_SPOT)

        # Draw base with glow
        pygame.gfxdraw.filled_circle(self.screen, int(self.base_pos[0]), int(self.base_pos[1]), 25, self.COLOR_BASE_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(self.base_pos[0]), int(self.base_pos[1]), 20, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, int(self.base_pos[0]), int(self.base_pos[1]), 20, self.COLOR_BASE)

        # Draw towers
        for tower in self.towers:
            spec = self.tower_types[tower['type']]
            pos = (int(tower['pos'][0]), int(tower['pos'][1]))
            pygame.draw.circle(self.screen, (100,100,100), pos, int(spec['range']), 1) # Range indicator
            if spec['name'] == "Cannon":
                pygame.draw.rect(self.screen, spec['color'], (pos[0]-10, pos[1]-10, 20, 20))
            elif spec['name'] == "Gatling":
                pygame.draw.polygon(self.screen, spec['color'], [(pos[0], pos[1]-12), (pos[0]-10, pos[1]+8), (pos[0]+10, pos[1]+8)])

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, (255,255,255))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, proj['color'])
            
        # Draw enemies with health bars
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_ENEMY_GLOW)
            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_ENEMY)
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            bar_width = 20
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (pos[0] - bar_width/2, pos[1] - 20, bar_width, 5))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (pos[0] - bar_width/2, pos[1] - 20, bar_width * health_pct, 5))

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

        # Draw cursor and tower preview
        cursor_pos = self.placement_spots[self.cursor_pos_index]
        tower_type = self.tower_types[self.selected_tower_type]
        is_occupied = any(t['pos'] == cursor_pos for t in self.towers)
        can_afford = self.money >= tower_type['cost']
        
        # Draw preview
        preview_color = (*tower_type['color'], 50) if can_afford and not is_occupied else (255, 0, 0, 50)
        pos_int = (int(cursor_pos[0]), int(cursor_pos[1]))
        if tower_type['name'] == "Cannon":
            pygame.draw.rect(self.screen, preview_color, (pos_int[0]-10, pos_int[1]-10, 20, 20))
        elif tower_type['name'] == "Gatling":
            pygame.draw.polygon(self.screen, preview_color, [(pos_int[0], pos_int[1]-12), (pos_int[0]-10, pos_int[1]+8), (pos_int[0]+10, pos_int[1]+8)])

        # Draw cursor
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_pos[0] - 18, cursor_pos[1] - 18, 36, 36), 2, 4)

    def _render_ui(self):
        # Top-left info: Wave and Base Health
        wave_text = self.font_large.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        health_text = self.font_small.render("BASE HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 40))
        health_pct = max(0, self.base_health / self.max_base_health)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 60, 150, 15))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 60, 150 * health_pct, 15))

        # Top-right info: Score and Money
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        money_text = self.font_large.render(f"$: {self.money}", True, self.COLOR_CURSOR)
        self.screen.blit(money_text, (self.WIDTH - money_text.get_width() - 10, 40))

        # Bottom-center info: Selected Tower
        tower_type = self.tower_types[self.selected_tower_type]
        tower_info = f"Tower: {tower_type['name']} | Cost: {tower_type['cost']}"
        tower_text = self.font_small.render(tower_info, True, self.COLOR_UI_TEXT)
        self.screen.blit(tower_text, (self.WIDTH/2 - tower_text.get_width()/2, self.HEIGHT - 25))

        # Game over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            msg_render = self.font_large.render(msg, True, (255, 255, 255))
            self.screen.blit(msg_render, (self.WIDTH/2 - msg_render.get_width()/2, self.HEIGHT/2 - msg_render.get_height()/2))


    def validate_implementation(self):
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
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        move = 0 # no-op
        if keys[pygame.K_UP]: move = 1
        elif keys[pygame.K_DOWN]: move = 2
        elif keys[pygame.K_LEFT]: move = 3
        elif keys[pygame.K_RIGHT]: move = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move, space, shift]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart.")

        clock.tick(env.FPS)
        
    pygame.quit()