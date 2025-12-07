import os
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector between tower slots. "
        "Hold Shift to cycle tower types. Press Space to build the selected tower."
    )

    game_description = (
        "A minimalist tower defense game. Place towers strategically to defend your base "
        "from waves of incoming enemies. Survive all waves to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 18000  # 10 minutes at 30 FPS

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_PATH = (45, 50, 62)
    COLOR_BASE = (68, 189, 114)
    COLOR_ENEMY = (224, 82, 99)
    COLOR_TOWER_1 = (255, 204, 0)
    COLOR_TOWER_2 = (0, 170, 255)
    COLOR_PROJECTILE_1 = (255, 224, 100)
    COLOR_PROJECTILE_2 = (100, 200, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_HEALTH = (68, 189, 114)
    COLOR_UI_HEALTH_BG = (80, 20, 20)
    COLOR_SELECTOR_VALID = (255, 255, 255, 100)
    COLOR_SELECTOR_INVALID = (224, 82, 99, 100)

    TOWER_SPECS = {
        1: {"cost": 100, "range": 80, "damage": 10, "fire_rate": 0.8, "color": COLOR_TOWER_1, "proj_color": COLOR_PROJECTILE_1, "name": "PULSAR"},
        2: {"cost": 150, "range": 150, "damage": 35, "fire_rate": 2.0, "color": COLOR_TOWER_2, "proj_color": COLOR_PROJECTILE_2, "name": "RAILGUN"},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.font_huge = pygame.font.Font(None, 72)
        
        self._define_level()
        self.reset()
        
        # This is a good practice to ensure the implementation is correct
        # self.validate_implementation() 

    def _define_level(self):
        self.path = [
            (-20, 200), (100, 200), (100, 100), (400, 100),
            (400, 300), (200, 300), (200, 200), (540, 200), (540, 50), (660, 50)
        ]
        self.path_lengths = [math.hypot(self.path[i+1][0] - self.path[i][0], self.path[i+1][1] - self.path[i][1]) for i in range(len(self.path)-1)]
        self.total_path_length = sum(self.path_lengths)
        
        self.tower_slots = [
            (150, 150), (350, 150), (150, 250), (450, 250), (300, 50), (300, 350), (500, 125)
        ]
        
        self.waves_config = [
            {"count": 5, "health": 100, "speed": 40, "interval": 1.0, "reward": 10},
            {"count": 8, "health": 120, "speed": 45, "interval": 0.8, "reward": 15},
            {"count": 12, "health": 140, "speed": 45, "interval": 0.7, "reward": 20},
            {"count": 15, "health": 150, "speed": 50, "interval": 0.6, "reward": 25},
            {"count": 1, "health": 2000, "speed": 35, "interval": 0.5, "reward": 50}, # Mini-boss
            {"count": 20, "health": 160, "speed": 55, "interval": 0.5, "reward": 30},
            {"count": 25, "health": 180, "speed": 60, "interval": 0.4, "reward": 35},
            {"count": 15, "health": 300, "speed": 50, "interval": 0.8, "reward": 40}, # Armored
            {"count": 30, "health": 150, "speed": 70, "interval": 0.3, "reward": 50}, # Fast swarm
            {"count": 3, "health": 3000, "speed": 40, "interval": 2.0, "reward": 150}, # Boss wave
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            random.seed(seed)
        else:
            self.np_random, _ = gym.utils.seeding.np_random()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.reward_this_step = 0

        self.base_health = 100
        self.max_base_health = 100
        self.resources = 250

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.wave_number = 0
        self.wave_spawning = False
        self.enemies_in_wave = 0
        self.spawn_timer = 0
        self.inter_wave_timer = 5.0 # Time before first wave

        self.selector_index = 0
        self.selected_tower_type = 1
        
        self.space_was_held = False
        self.shift_was_held = False
        self.move_cooldown = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = -0.001 # Small penalty for time passing
        self.steps += 1
        dt = 1 / self.FPS

        self._handle_input(action)
        
        if not self.game_over:
            self._update_waves(dt)
            self._update_towers(dt)
            self._update_enemies(dt)
            self._update_projectiles(dt)
        
        self._update_particles(dt)
        self._cleanup()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            if self.game_won:
                self.reward_this_step += 100
            else:
                self.reward_this_step -= 100
        
        self.score += self.reward_this_step

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Tower Type Selection ---
        if shift_held and not self.shift_was_held:
            self.selected_tower_type = (self.selected_tower_type % len(self.TOWER_SPECS)) + 1
            # sfx: UI_switch
        self.shift_was_held = shift_held

        # --- Selector Movement ---
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        
        if movement != 0 and self.move_cooldown == 0:
            if movement == 1: # Up
                self.selector_index = (self.selector_index - 1) % len(self.tower_slots)
            elif movement == 2: # Down
                self.selector_index = (self.selector_index + 1) % len(self.tower_slots)
            elif movement == 3: # Left
                self.selector_index = (self.selector_index - 1) % len(self.tower_slots)
            elif movement == 4: # Right
                self.selector_index = (self.selector_index + 1) % len(self.tower_slots)
            self.move_cooldown = 5 # 5 frames cooldown
            # sfx: UI_move_selector
            
        # --- Tower Placement ---
        if space_held and not self.space_was_held:
            slot_pos = self.tower_slots[self.selector_index]
            is_occupied = any(t['pos'] == slot_pos for t in self.towers)
            spec = self.TOWER_SPECS[self.selected_tower_type]
            
            if not is_occupied and self.resources >= spec['cost']:
                self.resources -= spec['cost']
                self.towers.append({
                    "pos": slot_pos, "type": self.selected_tower_type,
                    "cooldown": 0, "target": None, **spec
                })
                self.reward_this_step += 0.5 # Reward for building
                # sfx: Tower_place
                self._create_particles(slot_pos, spec['color'], 20, 5, 15)
        self.space_was_held = space_held
        
    def _update_waves(self, dt):
        if self.game_won: return

        if self.wave_spawning:
            self.spawn_timer -= dt
            if self.spawn_timer <= 0 and self.enemies_in_wave > 0:
                self.enemies_in_wave -= 1
                wave_data = self.waves_config[self.wave_number - 1]
                self.spawn_timer = wave_data['interval']
                
                # Scale stats based on wave number (beyond initial config)
                health_scale = 1 + (self.wave_number - 1) * 0.05
                speed_scale = 1 + (self.wave_number - 1) * 0.05
                
                self.enemies.append({
                    "pos": list(self.path[0]),
                    "health": wave_data['health'] * health_scale,
                    "max_health": wave_data['health'] * health_scale,
                    "speed": wave_data['speed'] * speed_scale,
                    "path_index": 0,
                    "dist_on_path": 0,
                    "id": self.steps + self.enemies_in_wave
                })
        elif not self.enemies and self.wave_number < len(self.waves_config):
            self.inter_wave_timer -= dt
            if self.inter_wave_timer <= 0:
                self.wave_spawning = True
                self.wave_number += 1
                wave_data = self.waves_config[self.wave_number - 1]
                self.enemies_in_wave = wave_data['count']
                self.spawn_timer = 0
                self.inter_wave_timer = 10.0 # Time between waves
        elif not self.enemies and self.wave_number == len(self.waves_config):
            self.game_won = True


    def _update_towers(self, dt):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - dt)
            
            # Find new target if needed
            valid_targets = [e for e in self.enemies if math.hypot(e['pos'][0] - tower['pos'][0], e['pos'][1] - tower['pos'][1]) <= tower['range']]
            if valid_targets:
                # Target enemy furthest along the path
                tower['target'] = max(valid_targets, key=lambda e: e['dist_on_path'])
            else:
                tower['target'] = None
                
            if tower['target'] and tower['cooldown'] <= 0:
                tower['cooldown'] = tower['fire_rate']
                self.projectiles.append({
                    "pos": list(tower['pos']),
                    "target_id": tower['target']['id'],
                    "damage": tower['damage'],
                    "speed": 300,
                    "color": tower['proj_color']
                })
                # sfx: Tower_fire
                self._create_particles(tower['pos'], tower['color'], 5, 2, 8, -math.pi, math.pi)

    def _update_enemies(self, dt):
        for enemy in self.enemies:
            if enemy['path_index'] >= len(self.path) - 1:
                continue

            dist_to_move = enemy['speed'] * dt
            enemy['dist_on_path'] += dist_to_move
            
            while dist_to_move > 0 and enemy['path_index'] < len(self.path) - 1:
                p1 = self.path[enemy['path_index']]
                p2 = self.path[enemy['path_index'] + 1]
                
                segment_vec = (p2[0] - p1[0], p2[1] - p1[1])
                segment_len = self.path_lengths[enemy['path_index']]
                
                current_pos_on_segment = (enemy['pos'][0] - p1[0], enemy['pos'][1] - p1[1])
                dist_on_segment = math.hypot(current_pos_on_segment[0], current_pos_on_segment[1])
                
                remaining_dist_on_segment = segment_len - dist_on_segment
                
                if dist_to_move >= remaining_dist_on_segment:
                    enemy['pos'] = list(p2)
                    enemy['path_index'] += 1
                    dist_to_move -= remaining_dist_on_segment
                else:
                    move_ratio = dist_to_move / segment_len if segment_len > 0 else 0
                    enemy['pos'][0] += segment_vec[0] * move_ratio
                    enemy['pos'][1] += segment_vec[1] * move_ratio
                    dist_to_move = 0

            if enemy['path_index'] >= len(self.path) - 1:
                self.base_health = max(0, self.base_health - 10)
                self.reward_this_step -= 10
                enemy['health'] = 0 # Mark for removal
                # sfx: Base_damage
                self._create_particles((self.SCREEN_WIDTH-50, self.SCREEN_HEIGHT/2), self.COLOR_ENEMY, 50, 10, 30)

    def _update_projectiles(self, dt):
        for p in self.projectiles:
            target = next((e for e in self.enemies if e['id'] == p['target_id']), None)
            
            if target:
                target_pos = target['pos']
                dx, dy = target_pos[0] - p['pos'][0], target_pos[1] - p['pos'][1]
                dist = math.hypot(dx, dy)
                
                if dist < 5: # Hit
                    target['health'] -= p['damage']
                    self.reward_this_step += 0.1
                    p['speed'] = 0 # Mark for removal
                    # sfx: Projectile_hit
                    self._create_particles(target['pos'], p['color'], 10, 3, 10)
                    if target['health'] <= 0:
                        wave_data = self.waves_config[self.wave_number-1]
                        self.resources += wave_data['reward']
                        self.reward_this_step += 2.0
                        # sfx: Enemy_die
                        self._create_particles(target['pos'], self.COLOR_ENEMY, 30, 5, 20)
                else:
                    p['pos'][0] += (dx / dist) * p['speed'] * dt
                    p['pos'][1] += (dy / dist) * p['speed'] * dt
            else:
                p['speed'] = 0 # Mark for removal if target is gone

    def _update_particles(self, dt):
        for particle in self.particles:
            particle['pos'][0] += particle['vel'][0] * dt
            particle['pos'][1] += particle['vel'][1] * dt
            particle['life'] -= dt
            particle['size'] = max(0, particle['size'] - 15 * dt)

    def _cleanup(self):
        self.enemies = [e for e in self.enemies if e['health'] > 0]
        self.projectiles = [p for p in self.projectiles if p['speed'] > 0]
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.game_won and not self.enemies:
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
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.wave_number,
            "game_over": self.game_over,
            "game_won": self.game_won,
        }

    def _render_game(self):
        # Draw Path
        for i in range(len(self.path) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path[i], self.path[i+1], 30)

        # Draw Base
        base_rect = pygame.Rect(self.SCREEN_WIDTH - 20, self.path[-1][1]-20, 40, 40)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

        # Draw Tower Slots and Selector
        for i, slot_pos in enumerate(self.tower_slots):
            is_occupied = any(t['pos'] == slot_pos for t in self.towers)
            color = (100, 100, 100, 50) if is_occupied else (200, 200, 200, 50)
            pygame.gfxdraw.filled_circle(self.screen, int(slot_pos[0]), int(slot_pos[1]), 20, color)
            pygame.gfxdraw.aacircle(self.screen, int(slot_pos[0]), int(slot_pos[1]), 20, color)

        # Draw Selector
        selector_pos = self.tower_slots[self.selector_index]
        is_occupied = any(t['pos'] == selector_pos for t in self.towers)
        spec = self.TOWER_SPECS[self.selected_tower_type]
        can_afford = self.resources >= spec['cost']
        selector_color = self.COLOR_SELECTOR_VALID if not is_occupied and can_afford else self.COLOR_SELECTOR_INVALID
        
        temp_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, int(selector_pos[0]), int(selector_pos[1]), 25, selector_color)
        pygame.gfxdraw.aacircle(temp_surf, int(selector_pos[0]), int(selector_pos[1]), 25, selector_color)
        self.screen.blit(temp_surf, (0,0))

        # Draw Towers
        for tower in self.towers:
            pos = (int(tower['pos'][0]), int(tower['pos'][1]))
            pygame.draw.rect(self.screen, tower['color'], (pos[0]-10, pos[1]-10, 20, 20))
            # Cooldown indicator
            cooldown_ratio = tower['cooldown'] / tower['fire_rate']
            if cooldown_ratio > 0:
                pygame.draw.arc(self.screen, (255,255,255), (pos[0]-12, pos[1]-12, 24, 24), 0, cooldown_ratio * 2 * math.pi, 2)
            # Range indicator for selected tower
            if tower['pos'] == self.tower_slots[self.selector_index]:
                 pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(tower['range']), (255,255,255,50))


        # Draw Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            pygame.draw.rect(self.screen, (50,0,0), (pos[0]-10, pos[1]-15, 20, 3))
            pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (pos[0]-10, pos[1]-15, 20 * health_ratio, 3))

        # Draw Projectiles
        for p in self.projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.line(self.screen, p['color'], pos, (pos[0], pos[1]+2), 3)

        # Draw Particles
        for particle in self.particles:
            pos = (int(particle['pos'][0]), int(particle['pos'][1]))
            size = int(particle['size'])
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, particle['color'])

    def _render_ui(self):
        # Health Bar
        health_ratio = self.base_health / self.max_base_health
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (10, 10, bar_width * health_ratio, 20))
        health_text = self.font_small.render(f"Base: {int(self.base_health)}/{self.max_base_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Resources
        res_text = self.font_large.render(f"${self.resources}", True, self.COLOR_TOWER_1)
        self.screen.blit(res_text, (10, 40))
        
        # Wave Info
        wave_str = f"Wave: {self.wave_number}/{len(self.waves_config)}"
        if self.game_won: wave_str = "VICTORY!"
        wave_text = self.font_large.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # Selected Tower Info
        spec = self.TOWER_SPECS[self.selected_tower_type]
        can_afford = self.resources >= spec['cost']
        tower_color = spec['color'] if can_afford else self.COLOR_ENEMY
        name_text = self.font_large.render(spec['name'], True, tower_color)
        cost_text = self.font_small.render(f"Cost: ${spec['cost']}", True, tower_color)
        self.screen.blit(name_text, (10, self.SCREEN_HEIGHT - 70))
        self.screen.blit(cost_text, (10, self.SCREEN_HEIGHT - 35))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY" if self.game_won else "GAME OVER"
            color = self.COLOR_BASE if self.game_won else self.COLOR_ENEMY
            end_text = self.font_huge.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, pos, color, count, min_speed, max_speed, angle_min=-math.pi, angle_max=math.pi):
        for _ in range(count):
            angle = self.np_random.uniform(angle_min, angle_max)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = [math.cos(angle) * speed * 10, math.sin(angle) * speed * 10]
            self.particles.append({
                "pos": list(pos), "vel": vel, "life": self.np_random.uniform(0.3, 0.8),
                "color": color, "size": self.np_random.uniform(2, 5)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Gymnasium environments are not typically run this way for training,
    # but it's useful for testing and visualization.
    # To run with a window, comment out the `os.environ` line at the top of the file.
    
    env = GameEnv()
    try:
        env.validate_implementation()
    except AssertionError as e:
        print(f"Validation failed: {e}")
        
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    # This will fail if the dummy driver is enabled.
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Tower Defense")
        clock = pygame.time.Clock()
        running = True
    except pygame.error:
        print("Could not create display. Running in headless mode. The __main__ block will not display anything.")
        running = False

    
    total_reward = 0
    
    while running:
        # --- Action mapping for human keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Won: {info['game_won']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()