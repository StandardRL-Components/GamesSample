
# Generated: 2025-08-28T03:38:47.368293
# Source Brief: brief_02083.md
# Brief Index: 2083

        
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
        "Controls: Arrow keys to move cursor. Space to place tower. Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers along the path."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 3000
    FPS = 30

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_PATH = (60, 60, 70)
    COLOR_BASE = (60, 180, 70)
    COLOR_PLACEMENT_SPOT = (45, 45, 55)
    COLOR_ENEMY = (210, 40, 40)
    COLOR_HEALTH_BAR_BG = (80, 20, 20)
    COLOR_HEALTH_BAR = (220, 60, 60)
    COLOR_TEXT = (240, 240, 240)
    COLOR_CURSOR_VALID = (100, 255, 100, 150)
    COLOR_CURSOR_INVALID = (255, 100, 100, 150)
    
    TOWER_SPECS = [
        {
            "name": "Cannon", "cost": 30, "range": 80, "damage": 10,
            "cooldown": 45, "color": (80, 150, 255), "proj_speed": 5
        },
        {
            "name": "Sniper", "cost": 60, "range": 150, "damage": 35,
            "cooldown": 120, "color": (255, 220, 80), "proj_speed": 10
        },
        {
            "name": "Machine Gun", "cost": 80, "range": 60, "damage": 5,
            "cooldown": 15, "color": (150, 255, 150), "proj_speed": 7
        },
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game-specific setup
        self.path = [
            (-20, 200), (80, 200), (80, 80), (320, 80), (320, 320),
            (560, 320), (560, 150), (self.SCREEN_WIDTH + 20, 150)
        ]
        self.placement_spots = [
            (140, 140), (260, 140), (150, 260), (390, 250), (490, 220), (490, 80)
        ]
        
        # Initialize state variables
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = np.array([0.0, 0.0])
        self.selected_tower_type = 0
        self.last_shift_press = False
        self.last_space_press = False
        
        self.np_random = None
        
        self.reset()
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.lives = 10
        self.gold = 100
        self.current_wave = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.wave_cooldown = self.FPS * 5 # 5 seconds before first wave
        self.enemies_to_spawn = 0
        self.enemy_spawn_timer = 0
        
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.selected_tower_type = 0
        self.last_shift_press = False
        self.last_space_press = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.game_over = self.lives <= 0
        terminated = self.game_over or self.game_won or self.steps >= self.MAX_STEPS

        if terminated:
            if self.game_won:
                reward += 100
            elif self.game_over:
                reward -= 100
            # Return final state
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Handle player input
        self._handle_input(action)

        # Update all game objects and collect rewards
        reward += self._update_game_state()
        
        self.steps += 1
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        cursor_speed = 8
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)
        
        if shift_held and not self.last_shift_press:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        self.last_shift_press = shift_held
        
        if space_held and not self.last_space_press:
            tower_spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.gold >= tower_spec["cost"]:
                for spot_pos in self.placement_spots:
                    dist = np.linalg.norm(self.cursor_pos - spot_pos)
                    if dist < 20 and not any(np.array_equal(t['pos'], spot_pos) for t in self.towers):
                        self.towers.append({
                            "pos": np.array(spot_pos, dtype=float),
                            "type": self.selected_tower_type,
                            "cooldown": 0,
                        })
                        self.gold -= tower_spec["cost"]
                        # sfx: tower_place()
                        break
        self.last_space_press = space_held

    def _update_game_state(self):
        step_reward = 0

        # Wave Management
        if self.enemies_to_spawn == 0 and not self.enemies:
            if self.current_wave > 0:
                step_reward += 50
            
            if self.current_wave == 10:
                self.game_won = True
                return step_reward
                
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0:
                self.current_wave += 1
                self.enemies_to_spawn = 5 + self.current_wave * 2
                self.enemy_spawn_timer = 0
                self.wave_cooldown = self.FPS * 15

        # Enemy Spawning
        if self.enemies_to_spawn > 0:
            self.enemy_spawn_timer -= 1
            if self.enemy_spawn_timer <= 0:
                self.enemy_spawn_timer = self.FPS * 0.5
                self.enemies_to_spawn -= 1
                
                health = 50 + (self.current_wave - 1) * 15 * (1 + (self.current_wave - 1) * 0.1)
                speed = 1.0 + (self.current_wave - 1) * 0.1
                
                self.enemies.append({
                    "pos": np.array(self.path[0], dtype=float), "health": health,
                    "max_health": health, "speed": speed, "path_index": 1,
                    "id": self.steps + self.np_random.integers(10000)
                })

        # Tower Logic
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            
            if tower['cooldown'] == 0:
                target = None
                min_dist = spec['range']
                for enemy in self.enemies:
                    dist = np.linalg.norm(tower['pos'] - enemy['pos'])
                    if dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    self.projectiles.append({
                        "pos": tower['pos'].copy(), "target_id": target['id'],
                        "speed": spec['proj_speed'], "damage": spec['damage'],
                        "color": spec['color']
                    })
                    tower['cooldown'] = spec['cooldown']
                    # sfx: tower_shoot()

        # Projectile Logic
        for proj in self.projectiles[:]:
            target_enemy = next((e for e in self.enemies if e['id'] == proj['target_id']), None)
            
            if not target_enemy:
                self.projectiles.remove(proj)
                continue

            direction = target_enemy['pos'] - proj['pos']
            dist = np.linalg.norm(direction)
            
            if dist < proj['speed']:
                target_enemy['health'] -= proj['damage']
                self._create_particles(target_enemy['pos'], proj['color'], 5, 2)
                self.projectiles.remove(proj)
                # sfx: enemy_hit()
            else:
                proj['pos'] += (direction / dist) * proj['speed']

        # Enemy Logic
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                gold_gain = 5 + self.current_wave
                self.gold += gold_gain
                step_reward += 1 + (gold_gain * 0.1)
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 15, 3)
                self.enemies.remove(enemy)
                # sfx: enemy_die()
                continue
                
            target_waypoint = self.path[enemy['path_index']]
            direction = target_waypoint - enemy['pos']
            dist = np.linalg.norm(direction)
            
            if dist < enemy['speed']:
                enemy['path_index'] += 1
                if enemy['path_index'] >= len(self.path):
                    self.lives -= 1
                    self.enemies.remove(enemy)
                    step_reward -= 25
                    self._create_particles(enemy['pos'], self.COLOR_BASE, 30, 5)
                    # sfx: life_lost()
                else:
                    enemy['pos'] = np.array(target_waypoint, dtype=float)
            else:
                enemy['pos'] += (direction / dist) * enemy['speed']

        # Particle Logic
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
                
        return step_reward

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
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 30)
        base_pos = self.path[-1]
        pygame.draw.circle(self.screen, self.COLOR_BASE, (int(base_pos[0]-20), int(base_pos[1])), 20)

        for spot in self.placement_spots:
            is_occupied = any(np.array_equal(t['pos'], spot) for t in self.towers)
            color = (80,80,90) if is_occupied else self.COLOR_PLACEMENT_SPOT
            pygame.gfxdraw.filled_circle(self.screen, int(spot[0]), int(spot[1]), 18, color)
            pygame.gfxdraw.aacircle(self.screen, int(spot[0]), int(spot[1]), 18, (color[0]+20, color[1]+20, color[2]+20))

        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            pos = (int(tower['pos'][0]), int(tower['pos'][1]))
            pygame.draw.circle(self.screen, spec['color'], pos, 15)
            pygame.draw.circle(self.screen, self.COLOR_BG, pos, 10)
            pygame.draw.circle(self.screen, spec['color'], pos, 5)
            if np.linalg.norm(self.cursor_pos - tower['pos']) < 30:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], spec['range'], (*spec['color'], 50))

        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, 8)
            health_ratio = enemy['health'] / enemy['max_health']
            bar_width = 16
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (pos[0] - bar_width/2, pos[1] - 15, bar_width, 4))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (pos[0] - bar_width/2, pos[1] - 15, bar_width * health_ratio, 4))
            
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, (*proj['color'], 150))
            pygame.draw.circle(self.screen, proj['color'], pos, 3)
            
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifetime'] / p['max_lifetime']))))
            size = int(p['size'] * (p['lifetime'] / p['max_lifetime']))
            if size > 0:
                color = (*p['color'], alpha)
                surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (size, size), size)
                self.screen.blit(surf, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

    def _render_ui(self):
        gold_text = self.font_small.render(f"GOLD: {self.gold}", True, (255, 223, 0))
        self.screen.blit(gold_text, (10, 10))
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH/2 - lives_text.get_width()/2, 10))
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/10", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))
        
        ui_box = pygame.Rect(self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 90, 140, 80)
        pygame.draw.rect(self.screen, (30, 30, 40), ui_box, border_radius=5)
        
        for i, spec in enumerate(self.TOWER_SPECS):
            x_offset = ui_box.x + 25 + i * 40
            y_offset = ui_box.y + 25
            if i == self.selected_tower_type:
                pygame.draw.circle(self.screen, (255, 255, 255), (x_offset, y_offset), 15, 2)
            pygame.draw.circle(self.screen, spec['color'], (x_offset, y_offset), 12)
        
        spec = self.TOWER_SPECS[self.selected_tower_type]
        name_text = self.font_small.render(f"{spec['name']}", True, self.COLOR_TEXT)
        cost_text = self.font_small.render(f"Cost: {spec['cost']}", True, (255, 223, 0))
        self.screen.blit(name_text, (ui_box.centerx - name_text.get_width()/2, ui_box.y + 45))
        self.screen.blit(cost_text, (ui_box.centerx - cost_text.get_width()/2, ui_box.y + 60))

        cursor_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
        is_valid_spot = False
        tower_spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.gold >= tower_spec["cost"]:
            for spot_pos in self.placement_spots:
                if np.linalg.norm(self.cursor_pos - spot_pos) < 20 and not any(np.array_equal(t['pos'], spot_pos) for t in self.towers):
                    is_valid_spot = True
                    break
        cursor_color = self.COLOR_CURSOR_VALID if is_valid_spot else self.COLOR_CURSOR_INVALID
        pygame.gfxdraw.aacircle(cursor_surf, 30, 30, 20, cursor_color)
        pygame.gfxdraw.filled_circle(cursor_surf, 30, 30, 20, (cursor_color[0], cursor_color[1], cursor_color[2], cursor_color[3]//3))
        self.screen.blit(cursor_surf, (int(self.cursor_pos[0]-30), int(self.cursor_pos[1]-30)))
        
        if self.game_over:
            text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY)
            self.screen.blit(text, (self.SCREEN_WIDTH/2 - text.get_width()/2, self.SCREEN_HEIGHT/2 - text.get_height()/2))
        elif self.game_won:
            text = self.font_large.render("YOU WIN!", True, self.COLOR_BASE)
            self.screen.blit(text, (self.SCREEN_WIDTH/2 - text.get_width()/2, self.SCREEN_HEIGHT/2 - text.get_height()/2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "lives": self.lives,
            "wave": self.current_wave,
            "enemies_remaining": len(self.enemies) + self.enemies_to_spawn,
            "towers_placed": len(self.towers)
        }

    def _create_particles(self, pos, color, count, speed_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifetime = self.np_random.integers(self.FPS//2, self.FPS)
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "color": color,
                "lifetime": lifetime, "max_lifetime": lifetime,
                "size": self.np_random.integers(2, 5)
            })
            
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        font = pygame.font.Font(None, 24)
        reward_text = font.render(f"Last Reward: {reward:.2f}", True, (255, 255, 255))
        total_reward_text = font.render(f"Total Reward: {total_reward:.2f}", True, (255, 255, 255))
        screen.blit(reward_text, (10, 30))
        screen.blit(total_reward_text, (10, 50))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)
        
    env.close()