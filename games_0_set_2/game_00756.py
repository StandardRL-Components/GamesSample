
# Generated: 2025-08-27T14:40:50.312013
# Source Brief: brief_00756.md
# Brief Index: 756

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to select a build location. "
        "Press Shift to cycle tower types. Press Space to build the selected tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down tower defense game. Survive 15 waves of enemies by strategically placing towers along their path."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 4500 # 150 seconds
        self.MAX_WAVES = 15
        
        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_PATH = (50, 50, 65)
        self.COLOR_BASE = (60, 220, 120)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_TOWER_1 = (50, 150, 255)
        self.COLOR_TOWER_2 = (255, 180, 50)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_SELECT_GLOW = (255, 255, 0)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game assets (defined procedurally)
        self.path_points = [
            pygame.Vector2(-50, 200),
            pygame.Vector2(150, 200),
            pygame.Vector2(150, 100),
            pygame.Vector2(450, 100),
            pygame.Vector2(450, 300),
            pygame.Vector2(self.WIDTH + 50, 300)
        ]
        self.tower_spots = [
            pygame.Vector2(100, 150),
            pygame.Vector2(200, 150),
            pygame.Vector2(400, 150),
            pygame.Vector2(400, 250),
        ]
        self.tower_definitions = [
            {"name": "Gatling", "cost": 50, "damage": 4, "range": 80, "fire_rate": 0.3, "color": self.COLOR_TOWER_1},
            {"name": "Cannon", "cost": 120, "damage": 25, "range": 110, "fire_rate": 1.5, "color": self.COLOR_TOWER_2},
        ]

        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = 100
        self.gold = 150
        
        self.current_wave = 0
        self.wave_active = False
        self.time_to_next_wave = self.FPS * 5 # 5 seconds
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.selected_tower_spot_idx = -1
        self.selected_tower_type_idx = 0
        self.last_shift_state = False
        self.last_space_state = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.clock.tick(self.FPS)
        reward = 0
        
        if not self.game_over:
            # 1. Handle Input
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            self._handle_input(movement, space_held, shift_held)

            # 2. Wave Management
            if not self.wave_active and self.current_wave < self.MAX_WAVES:
                self.time_to_next_wave -= 1
                if self.time_to_next_wave <= 0:
                    self._start_wave()
            
            # 3. Update Game Objects
            self._update_towers()
            killed_this_step = self._update_enemies()
            self._update_projectiles(killed_this_step)
            self._update_particles()
            
            # 4. Calculate Rewards
            for _ in range(sum(k['killed'] for k in killed_this_step)):
                reward += 1.0
                # _sfx_kill_enemy()
            
            for k in killed_this_step:
                self.gold += k['gold']
            
            # 5. Check Wave Completion
            if self.wave_active and not self.enemies:
                self.wave_active = False
                self.time_to_next_wave = self.FPS * 7 # 7 seconds between waves
                reward += 100
                self.score += 100
                if self.current_wave >= self.MAX_WAVES:
                    self.win = True
                    # _sfx_win_game()
        
        # 6. Check Termination Conditions
        self.steps += 1
        self.score += reward
        terminated = False
        if self.base_health <= 0:
            self.base_health = 0
            terminated = True
            reward -= 100
            # _sfx_lose_game()
        elif self.win:
            terminated = True
            reward += 500
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Select tower spot with arrow keys
        if 1 <= movement <= 4:
            self.selected_tower_spot_idx = movement - 1
        
        # Cycle tower type on Shift press (rising edge)
        if shift_held and not self.last_shift_state:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.tower_definitions)
            # _sfx_ui_select()
        self.last_shift_state = shift_held

        # Place tower on Space press (rising edge)
        if space_held and not self.last_space_state and self.selected_tower_spot_idx != -1:
            spot_pos = self.tower_spots[self.selected_tower_spot_idx]
            tower_def = self.tower_definitions[self.selected_tower_type_idx]
            
            is_spot_occupied = any(t['pos'] == spot_pos for t in self.towers)
            
            if not is_spot_occupied and self.gold >= tower_def['cost']:
                self.gold -= tower_def['cost']
                self.towers.append({
                    "pos": spot_pos,
                    "type_idx": self.selected_tower_type_idx,
                    "cooldown": 0,
                    "target": None
                })
                # _sfx_place_tower()
        self.last_space_state = space_held

    def _start_wave(self):
        self.current_wave += 1
        self.wave_active = True
        
        num_enemies = 3 + (self.current_wave - 1) * 2
        enemy_health = 20 * (1.05 ** (self.current_wave - 1))
        enemy_speed = 1.5 * (1.05 ** (self.current_wave - 1))
        
        for i in range(num_enemies):
            self.enemies.append({
                "pos": self.path_points[0] - pygame.Vector2(i * 30, 0),
                "health": enemy_health,
                "max_health": enemy_health,
                "speed": enemy_speed,
                "path_idx": 1,
                "id": self.np_random.integers(1, 1_000_000)
            })

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            tower_def = self.tower_definitions[tower['type_idx']]
            
            if tower['cooldown'] <= 0:
                # Find a new target if current one is dead or out of range
                if tower.get('target'):
                    target_enemy = next((e for e in self.enemies if e['id'] == tower['target']), None)
                    if not target_enemy or tower['pos'].distance_to(target_enemy['pos']) > tower_def['range']:
                        tower['target'] = None
                
                # Acquire new target if needed
                if not tower.get('target'):
                    possible_targets = [e for e in self.enemies if tower['pos'].distance_to(e['pos']) <= tower_def['range']]
                    if possible_targets:
                        tower['target'] = possible_targets[0]['id']

                # Fire if target is valid
                target_enemy = next((e for e in self.enemies if e.get('id') == tower.get('target')), None)
                if target_enemy:
                    self.projectiles.append({
                        "pos": tower['pos'].copy(),
                        "target_id": target_enemy['id'],
                        "speed": 10,
                        "damage": tower_def['damage']
                    })
                    tower['cooldown'] = self.FPS * tower_def['fire_rate']
                    # _sfx_tower_fire()

    def _update_enemies(self):
        killed_this_step = []
        for enemy in reversed(self.enemies):
            if enemy['path_idx'] >= len(self.path_points):
                self.base_health -= 10
                self.enemies.remove(enemy)
                self._create_particles(enemy['pos'], self.COLOR_BASE, 15, 20)
                # _sfx_base_damage()
                continue

            target_pos = self.path_points[enemy['path_idx']]
            direction = (target_pos - enemy['pos']).normalize()
            enemy['pos'] += direction * enemy['speed']

            if enemy['pos'].distance_to(target_pos) < enemy['speed']:
                enemy['path_idx'] += 1
        
        return killed_this_step

    def _update_projectiles(self, killed_list):
        for p in reversed(self.projectiles):
            target_enemy = next((e for e in self.enemies if e['id'] == p['target_id']), None)
            
            if not target_enemy:
                self.projectiles.remove(p)
                continue

            direction = (target_enemy['pos'] - p['pos']).normalize()
            p['pos'] += direction * p['speed']

            if p['pos'].distance_to(target_enemy['pos']) < p['speed']:
                target_enemy['health'] -= p['damage']
                self.projectiles.remove(p)
                self._create_particles(p['pos'], self.COLOR_PROJECTILE, 3, 5)
                # _sfx_enemy_hit()

                if target_enemy['health'] <= 0:
                    killed_list.append({'killed': True, 'gold': 10})
                    self._create_particles(target_enemy['pos'], self.COLOR_ENEMY, 20, 30)
                    self.enemies.remove(target_enemy)


    def _update_particles(self):
        for particle in reversed(self.particles):
            particle['life'] -= 1
            particle['radius'] += particle['growth']
            if particle['life'] <= 0:
                self.particles.remove(particle)

    def _create_particles(self, pos, color, count, lifetime):
        for _ in range(count):
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
                "life": self.np_random.integers(lifetime // 2, lifetime),
                "radius": self.np_random.uniform(1, 3),
                "growth": self.np_random.uniform(0.1, 0.3),
                "color": color
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, [tuple(p) for p in self.path_points], 20)
        
        # Render Base (end of path)
        base_pos = self.path_points[-2]
        pygame.draw.rect(self.screen, self.COLOR_BASE, (base_pos.x, base_pos.y - 25, 20, 50))

        # Render Tower Placement Spots
        for i, spot in enumerate(self.tower_spots):
            is_occupied = any(t['pos'] == spot for t in self.towers)
            color = (100, 100, 100) if is_occupied else (80, 80, 80)
            pygame.draw.circle(self.screen, color, (int(spot.x), int(spot.y)), 20)
            if i == self.selected_tower_spot_idx and not is_occupied:
                glow_alpha = 128 + 127 * math.sin(self.steps * 0.2)
                pygame.gfxdraw.filled_circle(self.screen, int(spot.x), int(spot.y), 22, (*self.COLOR_SELECT_GLOW, int(glow_alpha)))

        # Render Towers
        for tower in self.towers:
            tower_def = self.tower_definitions[tower['type_idx']]
            pos = (int(tower['pos'].x), int(tower['pos'].y))
            if tower['type_idx'] == 0: # Gatling
                pygame.draw.polygon(self.screen, tower_def['color'], [
                    (pos[0], pos[1] - 12), (pos[0] - 10, pos[1] + 8), (pos[0] + 10, pos[1] + 8)
                ])
            else: # Cannon
                pygame.draw.rect(self.screen, tower_def['color'], (pos[0] - 10, pos[1] - 10, 20, 20))
            pygame.draw.circle(self.screen, (0,0,0,0), pos, int(tower_def['range']), 1) # Range indicator (for debug)

        # Render Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, 8)
            # Health bar
            health_pct = max(0, enemy['health'] / enemy['max_health'])
            pygame.draw.rect(self.screen, (255,0,0), (pos[0]-10, pos[1]-15, 20, 3))
            pygame.draw.rect(self.screen, (0,255,0), (pos[0]-10, pos[1]-15, int(20*health_pct), 3))

        # Render Projectiles
        for p in self.projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(p['pos'].x), int(p['pos'].y)), 3)

        # Render Particles
        for particle in self.particles:
            alpha = max(0, 255 * (particle['life'] / 20))
            color = (*particle['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(particle['pos'].x), int(particle['pos'].y), int(particle['radius']), color)
            particle['pos'] += particle['vel']

        # Render UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Health
        self._draw_text(f"Base Health: {int(self.base_health)}", (10, 10), font=self.font_medium)
        # Gold
        self._draw_text(f"Gold: {self.gold}", (self.WIDTH - 10, 10), font=self.font_medium, align="topright")
        # Wave
        wave_str = f"Wave: {self.current_wave}/{self.MAX_WAVES}" if self.current_wave > 0 else "Get Ready!"
        self._draw_text(wave_str, (self.WIDTH // 2, 10), font=self.font_medium, align="midtop")

        # Selected Tower Info
        tower_def = self.tower_definitions[self.selected_tower_type_idx]
        color = self.COLOR_UI_TEXT if self.gold >= tower_def['cost'] else self.COLOR_ENEMY
        self._draw_text(f"Selected: {tower_def['name']}", (10, self.HEIGHT - 35), font=self.font_small)
        self._draw_text(f"Cost: {tower_def['cost']}", (10, self.HEIGHT - 18), font=self.font_small, color=color)

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_BASE if self.win else self.COLOR_ENEMY
            self._draw_text(msg, (self.WIDTH // 2, self.HEIGHT // 2), font=self.font_large, color=color, align="center")

    def _draw_text(self, text, pos, font, color=(230, 230, 230), align="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        setattr(text_rect, align, pos)
        self.screen.blit(text_surface, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "gold": self.gold,
            "base_health": self.base_health,
        }

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy' or 'windib' depending on your system

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # --- Human Controls ---
    # `actions` is a list: [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    # space: 0=released, 1=held
    # shift: 0=released, 1=held
    action = [0, 0, 0] 

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Get key presses
        keys = pygame.key.get_pressed()
        
        # Movement mapping (map arrows to tower spots 1-4)
        # For this game, let's map them directly to spots 1-4 for simplicity
        action[0] = 0 # No-op
        if keys[pygame.K_UP]:
            action[0] = 1 # Spot 1
        elif keys[pygame.K_RIGHT]:
            action[0] = 2 # Spot 2
        elif keys[pygame.K_DOWN]:
            action[0] = 3 # Spot 3
        elif keys[pygame.K_LEFT]:
            action[0] = 4 # Spot 4

        # Space and Shift mapping
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0
            # Optional: pause on game over
            # running = False 

        clock.tick(env.FPS)
        
    pygame.quit()