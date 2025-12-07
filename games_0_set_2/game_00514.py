
# Generated: 2025-08-27T13:53:12.825850
# Source Brief: brief_00514.md
# Brief Index: 514

        
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
        "Controls: ↑↓ to select placement slot, ←→ to select tower type. Space to build, Shift to sell."
    )

    game_description = (
        "Defend your base from waves of geometric enemies by strategically placing various towers along their path."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.width, self.height = 640, 400

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()

        # --- Colors and Fonts ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_BASE = (50, 200, 100)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_HEALTH_GREEN = (40, 220, 110)
        self.COLOR_HEALTH_RED = (220, 40, 80)

        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # --- Game Constants ---
        self.MAX_STEPS = 30000 # 1000s @ 30fps
        self.STARTING_HEALTH = 100
        self.STARTING_MONEY = 250
        self.TOTAL_WAVES = 10
        self.BUILD_PHASE_DURATION = 10 * 30 # 10 seconds

        # --- Game Path ---
        self.path_points = [
            (-50, 150), (100, 150), (100, 300), (300, 300), (300, 100),
            (500, 100), (500, 250), (self.width + 50, 250)
        ]
        
        # --- Tower Placement Slots ---
        self.tower_slots = [
            pygame.Rect(120, 170, 40, 40), pygame.Rect(170, 170, 40, 40), pygame.Rect(220, 170, 40, 40),
            pygame.Rect(120, 240, 40, 40), pygame.Rect(170, 240, 40, 40), pygame.Rect(220, 240, 40, 40),
            pygame.Rect(320, 120, 40, 40), pygame.Rect(370, 120, 40, 40), pygame.Rect(420, 120, 40, 40),
            pygame.Rect(320, 190, 40, 40), pygame.Rect(370, 190, 40, 40), pygame.Rect(420, 190, 40, 40)
        ]

        # --- Tower Definitions ---
        self.tower_types = [
            {"name": "Cannon", "cost": 100, "damage": 10, "range": 80, "fire_rate": 45, "color": (50, 150, 255)},
            {"name": "Missile", "cost": 250, "damage": 50, "range": 120, "fire_rate": 90, "color": (255, 200, 50)},
            {"name": "Laser", "cost": 150, "damage": 3, "range": 70, "fire_rate": 10, "color": (200, 50, 255)},
            {"name": "Slower", "cost": 75, "damage": 1, "range": 90, "fire_rate": 30, "color": (50, 255, 200), "slow_factor": 0.5, "slow_duration": 60}
        ]

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = self.STARTING_HEALTH
        self.money = self.STARTING_MONEY
        self.wave_number = 0
        
        self.towers = {} # {slot_index: tower_dict}
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.game_phase = "BUILD"
        self.wave_timer = self.BUILD_PHASE_DURATION
        self.wave_spawn_timer = 0
        self.wave_enemies_to_spawn = []

        self.cursor_slot_index = 0
        self.selected_tower_type_index = 0
        
        self.input_cooldown = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.input_cooldown = max(0, self.input_cooldown - 1)
        
        reward += self._handle_input(action)
        
        if self.game_phase == "BUILD":
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_next_wave()
        
        elif self.game_phase == "WAVE":
            reward += self._update_wave()
            if not self.enemies and not self.wave_enemies_to_spawn:
                self.game_phase = "BUILD"
                self.wave_timer = self.BUILD_PHASE_DURATION
                reward += 1.0 # Wave survival bonus
                self.score += 100 * self.wave_number
                self.money += 100 + self.wave_number * 10

                if self.wave_number >= self.TOTAL_WAVES:
                    self.win = True
                    self.game_over = True

        self._update_particles()

        if self.base_health <= 0:
            self.game_over = True
            reward = -100
        elif self.win:
            reward = 100
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        terminated = self.game_over

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

        if self.input_cooldown == 0:
            if movement in [1, 2]: # Up/Down
                self.cursor_slot_index = (self.cursor_slot_index + (1 if movement == 2 else -1)) % len(self.tower_slots)
                self.input_cooldown = 5
            elif movement in [3, 4]: # Left/Right
                self.selected_tower_type_index = (self.selected_tower_type_index + (1 if movement == 4 else -1)) % len(self.tower_types)
                self.input_cooldown = 5
        
        # Build tower on space press
        if space_held and not self.last_space_held:
            tower_def = self.tower_types[self.selected_tower_type_index]
            if self.cursor_slot_index not in self.towers and self.money >= tower_def["cost"]:
                self.money -= tower_def["cost"]
                self.towers[self.cursor_slot_index] = {
                    "type": self.selected_tower_type_index,
                    "cooldown": 0,
                    "pos": self.tower_slots[self.cursor_slot_index].center
                }
                self._create_particles(self.towers[self.cursor_slot_index]['pos'], tower_def['color'], 20)
                # sfx: tower_place
        
        # Sell tower on shift press
        if shift_held and not self.last_shift_held:
            if self.cursor_slot_index in self.towers:
                tower_def = self.tower_types[self.towers[self.cursor_slot_index]["type"]]
                self.money += int(tower_def["cost"] * 0.75)
                pos = self.towers[self.cursor_slot_index]['pos']
                del self.towers[self.cursor_slot_index]
                self._create_particles(pos, (255, 255, 0), 15)
                # sfx: tower_sell

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        return reward

    def _start_next_wave(self):
        self.game_phase = "WAVE"
        self.wave_number += 1
        
        num_enemies = 5 + (self.wave_number - 1) * 2
        enemy_health = 20 + (self.wave_number - 1) * 10
        enemy_speed = 1.0 + self.wave_number * 0.05
        
        self.wave_enemies_to_spawn = []
        for i in range(num_enemies):
            self.wave_enemies_to_spawn.append({
                "health": enemy_health,
                "max_health": enemy_health,
                "speed": enemy_speed,
                "spawn_delay": i * 20 # Stagger spawns
            })
        self.wave_spawn_timer = 0
    
    def _update_wave(self):
        reward = 0
        
        # Spawn enemies
        self.wave_spawn_timer += 1
        if self.wave_enemies_to_spawn and self.wave_spawn_timer >= self.wave_enemies_to_spawn[0]["spawn_delay"]:
            enemy_data = self.wave_enemies_to_spawn.pop(0)
            self.enemies.append({
                "pos": pygame.math.Vector2(self.path_points[0]),
                "path_index": 1,
                "slow_timer": 0,
                **enemy_data
            })

        self._update_towers()
        reward += self._update_enemies()
        self._update_projectiles()
        
        return reward

    def _update_towers(self):
        for slot_index, tower in self.towers.items():
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                tower_def = self.tower_types[tower['type']]
                target = None
                min_dist = tower_def['range']
                
                for enemy in self.enemies:
                    dist = pygame.math.Vector2(tower['pos']).distance_to(enemy['pos'])
                    if dist <= min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    self.projectiles.append({
                        "start_pos": pygame.math.Vector2(tower['pos']),
                        "pos": pygame.math.Vector2(tower['pos']),
                        "target": target,
                        "tower_type": tower['type'],
                        "speed": 8
                    })
                    tower['cooldown'] = tower_def['fire_rate']
                    # sfx: tower_shoot

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy['slow_timer'] > 0:
                enemy['slow_timer'] -= 1
                current_speed = enemy['speed'] * 0.5
            else:
                current_speed = enemy['speed']
            
            if enemy['path_index'] < len(self.path_points):
                target_pos = pygame.math.Vector2(self.path_points[enemy['path_index']])
                direction = (target_pos - enemy['pos']).normalize()
                enemy['pos'] += direction * current_speed
                
                if enemy['pos'].distance_to(target_pos) < 5:
                    enemy['path_index'] += 1
            else:
                self.base_health -= 10
                self.score = max(0, self.score - 50)
                self.enemies.remove(enemy)
                self._create_particles(pygame.math.Vector2(self.width-20, 250), self.COLOR_HEALTH_RED, 30)
                # sfx: base_damage
                continue

            if enemy['health'] <= 0:
                reward += 0.1
                self.score += 10
                self.money += 5
                self._create_particles(enemy['pos'], (255, 120, 0), 25)
                self.enemies.remove(enemy)
                # sfx: enemy_explode
        return reward

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            if p['target'] not in self.enemies:
                self.projectiles.remove(p)
                continue
            
            target_pos = p['target']['pos']
            direction = (target_pos - p['pos']).normalize() if (target_pos - p['pos']).length() > 0 else pygame.math.Vector2(0,0)
            p['pos'] += direction * p['speed']
            
            if p['pos'].distance_to(target_pos) < 5:
                tower_def = self.tower_types[p['tower_type']]
                p['target']['health'] -= tower_def['damage']
                if "slow_factor" in tower_def:
                    p['target']['slow_timer'] = tower_def['slow_duration']
                self._create_particles(p['pos'], tower_def['color'], 5, 1)
                self.projectiles.remove(p)
                # sfx: projectile_hit

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count, speed_mult=1):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'lifespan': random.randint(15, 30),
                'color': color,
                'size': random.uniform(2, 5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_points, 30)
        
        # Base
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.width - 20, 225, 20, 50))
        
        # Tower slots and cursor
        for i, slot in enumerate(self.tower_slots):
            color = (255, 255, 255, 10) if i not in self.towers else (0,0,0,10)
            pygame.gfxdraw.box(self.screen, slot, color)
        cursor_rect = self.tower_slots[self.cursor_slot_index]
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)
        
        # Towers
        for i, tower in self.towers.items():
            tower_def = self.tower_types[tower['type']]
            rect = self.tower_slots[i]
            pygame.draw.rect(self.screen, tower_def['color'], rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect.inflate(-8, -8))

        # Projectiles
        for p in self.projectiles:
            tower_def = self.tower_types[p['tower_type']]
            pygame.draw.circle(self.screen, tower_def['color'], (int(p['pos'].x), int(p['pos'].y)), 3)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            color = (255, 70, 70)
            if enemy['slow_timer'] > 0:
                color = (100, 150, 255)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, color)
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            bar_w = 16
            pygame.draw.rect(self.screen, (50,50,50), (pos[0]-bar_w/2, pos[1]-15, bar_w, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (pos[0]-bar_w/2, pos[1]-15, int(bar_w*health_pct), 3))

        # Particles
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            size = int(p['size'])
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, p['color'])

    def _render_ui(self):
        # Top-left info: Wave and Timer
        if self.game_phase == "BUILD" and not self.win:
            wave_text = f"Wave {self.wave_number + 1} starts in {math.ceil(self.wave_timer / 30)}s"
        elif self.win:
            wave_text = "YOU WIN!"
        elif self.game_over and not self.win:
            wave_text = "GAME OVER"
        else:
            wave_text = f"Wave {self.wave_number} / {self.TOTAL_WAVES}"
        
        text_surf = self.font_m.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Top-right info: Health
        health_text = self.font_m.render(f"Base: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.width - health_text.get_width() - 10, 10))
        health_pct = self.base_health / self.STARTING_HEALTH
        health_bar_rect = pygame.Rect(self.width - 110, 40, 100, 10)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, health_bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (health_bar_rect.x, health_bar_rect.y, health_bar_rect.width * health_pct, health_bar_rect.height))

        # Bottom-left info: Tower selection
        sel_tower = self.tower_types[self.selected_tower_type_index]
        pygame.draw.rect(self.screen, sel_tower['color'], (10, self.height - 70, 60, 60))
        pygame.draw.rect(self.screen, self.COLOR_BG, (14, self.height - 66, 52, 52))
        
        name_surf = self.font_m.render(sel_tower['name'], True, sel_tower['color'])
        self.screen.blit(name_surf, (80, self.height - 70))
        
        cost_surf = self.font_s.render(f"Cost: {sel_tower['cost']}", True, self.COLOR_TEXT)
        self.screen.blit(cost_surf, (80, self.height - 45))
        
        dmg_surf = self.font_s.render(f"Dmg: {sel_tower['damage']} / Rng: {sel_tower['range']}", True, self.COLOR_TEXT)
        self.screen.blit(dmg_surf, (80, self.height - 25))

        # Bottom-right info: Score and Money
        score_text = self.font_m.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.width - score_text.get_width() - 10, self.height - 70))
        
        money_text = self.font_m.render(f"${self.money}", True, (255, 220, 100))
        self.screen.blit(money_text, (self.width - money_text.get_width() - 10, self.height - 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "money": self.money,
            "wave": self.wave_number,
            "base_health": self.base_health,
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
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.width, env.height))

    print(env.user_guide)
    print(env.game_description)

    action = env.action_space.sample()
    action.fill(0)

    while not done:
        # Human input mapping
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                if event.key == pygame.K_SPACE: action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]: action[0] = 0
                if event.key == pygame.K_SPACE: action[1] = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS

    env.close()