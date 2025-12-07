import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to select a build location. Press space to build the selected tower. Hold shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in this top-down tower defense game."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.rng = np.random.default_rng()

        # --- Colors and Fonts ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_BASE = (0, 150, 50)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_INVALID = (255, 0, 0, 100)

        self.FONT_UI = pygame.font.SysFont("Arial", 20, bold=True)
        self.FONT_POPUP = pygame.font.SysFont("Arial", 16)
        self.FONT_GAMEOVER = pygame.font.SysFont("Arial", 48, bold=True)

        # --- Game Definitions ---
        self.path_waypoints = [
            (-50, 200), (100, 200), (100, 80), (540, 80), (540, 320), (100, 320), (100, 200), (self.WIDTH + 50, 200)
        ]
        self.build_sites = [
            (180, 140), (280, 140), (380, 140), (480, 140),
            (180, 260), (280, 260), (380, 260), (480, 260),
        ]
        self.TOWER_TYPES = [
            {'name': 'Cannon', 'cost': 50, 'range': 80, 'damage': 10, 'fire_rate': 0.8, 'color': (60, 120, 255)},
            {'name': 'Missile', 'cost': 125, 'range': 120, 'damage': 35, 'fire_rate': 2.0, 'color': (255, 140, 0)}
        ]
        self.MAX_WAVES = 10
        self.MAX_STEPS = 15000 # ~8 minutes at 30fps

        # The initial reset is called to set up the game state.
        # It's good practice to have the environment ready after __init__.
        # The validation call has been removed from __init__ to avoid issues
        # with environments that might not be fully set up for validation yet.
        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Game State
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.base_health = 100
        self.resources = 150
        self.reward_this_step = 0.0 # Initialize reward attribute

        # Wave Management
        self.wave_number = 0
        self.wave_spawning_finished = False
        self.wave_enemy_count = 0
        self.wave_spawn_timer = 0
        self.intermission_timer = 150 # 5 seconds

        # Entities
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        # Player Input State
        self.cursor_index = 0
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.screen_flash_timer = 0

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.clock.tick(30)
        self.reward_this_step = -0.001 # Small penalty for time passing

        if self.game_over or self.victory:
            # Ensure final observation is correct and return
            obs = self._get_observation()
            # On termination, reward is handled by _check_termination, so we return 0 here to avoid double counting
            # if the function is called again after termination.
            return obs, 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        self._handle_input(movement, space_pressed, shift_pressed)
        
        self._update_wave_spawner()
        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        if self.screen_flash_timer > 0:
            self.screen_flash_timer -= 1
            
        self.steps += 1
        
        terminated = self._check_termination()
        self.score += self.reward_this_step

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        if movement != 0:
            # Simple grid navigation: up/down changes row, left/right changes column
            current_x, current_y = self.build_sites[self.cursor_index]
            
            best_site = -1
            min_dist = float('inf')

            for i, site in enumerate(self.build_sites):
                if i == self.cursor_index: continue
                dx, dy = site[0] - current_x, site[1] - current_y
                
                # Check if the site is in the general direction of movement
                is_dir_match = False
                if movement == 1 and dy < 0 and abs(dy) > abs(dx): is_dir_match = True # Up
                if movement == 2 and dy > 0 and abs(dy) > abs(dx): is_dir_match = True # Down
                if movement == 3 and dx < 0 and abs(dx) > abs(dy): is_dir_match = True # Left
                if movement == 4 and dx > 0 and abs(dx) > abs(dy): is_dir_match = True # Right

                if is_dir_match:
                    dist = math.hypot(dx, dy)
                    if dist < min_dist:
                        min_dist = dist
                        best_site = i
            
            if best_site != -1:
                self.cursor_index = best_site

        if shift_pressed:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
            # sfx: ui_cycle

        if space_pressed:
            self._place_tower()

    def _place_tower(self):
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        pos = self.build_sites[self.cursor_index]

        is_occupied = any(t['pos'] == pos for t in self.towers)
        if not is_occupied and self.resources >= tower_def['cost']:
            self.resources -= tower_def['cost']
            self.towers.append({
                'pos': pos,
                'type': self.selected_tower_type,
                'cooldown': 0,
                'target': None
            })
            # sfx: build_tower
            self._create_particles(pos, tower_def['color'], 20, 5, 20)
        else:
            # sfx: error
            pass
            
    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            self.victory = True
            return
        
        self.reward_this_step += 10.0 # Reward for starting a new wave
        
        # Difficulty scaling
        health_mult = 1 + (self.wave_number - 1) * 0.15
        speed_mult = 1 + (self.wave_number - 1) * 0.05
        
        self.wave_enemy_count = 10 + self.wave_number * 2
        self.wave_spawn_timer = 0
        self.wave_spawning_finished = False
        
        self.wave_definition = {
            'count': self.wave_enemy_count,
            'health': 20 * health_mult,
            'speed': 1.0 * speed_mult,
            'spawn_delay': max(10, 45 - self.wave_number * 2),
            'value': int(2 * health_mult)
        }

    def _update_wave_spawner(self):
        if self.wave_spawning_finished or self.game_over or self.victory:
            # Check for wave clear
            if not self.enemies and self.wave_spawning_finished and not self.victory:
                if self.intermission_timer > 0:
                    self.intermission_timer -= 1
                else:
                    self.resources += 100 + self.wave_number * 10
                    self.intermission_timer = 150 # Reset for next intermission
                    self._start_next_wave()
            return

        self.wave_spawn_timer -= 1
        if self.wave_spawn_timer <= 0 and self.wave_enemy_count > 0:
            self.wave_spawn_timer = self.wave_definition['spawn_delay']
            self.wave_enemy_count -= 1
            self.enemies.append({
                'pos': list(self.path_waypoints[0]),
                'health': self.wave_definition['health'],
                'max_health': self.wave_definition['health'],
                'speed': self.wave_definition['speed'],
                'path_index': 0,
                'dist_on_segment': 0,
                'value': self.wave_definition['value']
            })
            if self.wave_enemy_count == 0:
                self.wave_spawning_finished = True

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            path_idx = enemy['path_index']
            if path_idx >= len(self.path_waypoints) - 1:
                self.enemies.remove(enemy)
                self.base_health = max(0, self.base_health - 10)
                self.reward_this_step -= 5.0
                self.screen_flash_timer = 5
                # sfx: base_damage
                if self.base_health <= 0:
                    self.game_over = True
                continue
            
            start_pos = self.path_waypoints[path_idx]
            end_pos = self.path_waypoints[path_idx + 1]
            segment_vec = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
            segment_len = math.hypot(*segment_vec)
            
            enemy['dist_on_segment'] += enemy['speed']
            
            if enemy['dist_on_segment'] >= segment_len:
                enemy['path_index'] += 1
                enemy['dist_on_segment'] = 0
            else:
                progress = enemy['dist_on_segment'] / segment_len if segment_len > 0 else 0
                enemy['pos'][0] = start_pos[0] + segment_vec[0] * progress
                enemy['pos'][1] = start_pos[1] + segment_vec[1] * progress

    def _update_towers(self):
        for tower in self.towers:
            tower_def = self.TOWER_TYPES[tower['type']]
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1 / 30.0
                continue
            
            # Find a target
            target = None
            min_dist = float('inf')
            for enemy in self.enemies:
                dist = math.hypot(enemy['pos'][0] - tower['pos'][0], enemy['pos'][1] - tower['pos'][1])
                if dist <= tower_def['range'] and dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                tower['cooldown'] = tower_def['fire_rate']
                # sfx: shoot
                self.projectiles.append({
                    'pos': list(tower['pos']),
                    'type': tower['type'],
                    'target': target,
                    'damage': tower_def['damage']
                })
                self._create_particles(tower['pos'], tower_def['color'], 3, 2, 5, is_muzzle_flash=True)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = proj['target']['pos']
            proj_pos = proj['pos']
            
            move_vec = (target_pos[0] - proj_pos[0], target_pos[1] - proj_pos[1])
            dist = math.hypot(*move_vec)
            
            speed = 8 # Cannon speed
            if proj['type'] == 1: speed = 6 # Missile speed
            
            if dist < speed:
                self._handle_hit(proj)
                self.projectiles.remove(proj)
            else:
                proj_pos[0] += (move_vec[0] / dist) * speed
                proj_pos[1] += (move_vec[1] / dist) * speed
    
    def _handle_hit(self, projectile):
        target = projectile['target']
        if target not in self.enemies: return

        # sfx: hit_enemy
        self.reward_this_step += 0.1
        target['health'] -= projectile['damage']
        
        tower_color = self.TOWER_TYPES[projectile['type']]['color']
        self._create_particles(target['pos'], tower_color, 10, 3, 15)

        if target['health'] <= 0:
            self.resources += target['value']
            self.reward_this_step += 1.0
            self._create_particles(target['pos'], self.COLOR_ENEMY, 25, 4, 25)
            # sfx: enemy_destroyed
            self.enemies.remove(target)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count, speed_max, lifetime, is_muzzle_flash=False):
        for _ in range(count):
            if is_muzzle_flash:
                angle = self.rng.random() * 2 * math.pi
                speed = self.rng.random() * speed_max
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            else:
                vel = [(self.rng.random() - 0.5) * speed_max * 2, (self.rng.random() - 0.7) * speed_max * 2]
            
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'lifetime': self.rng.integers(lifetime // 2, lifetime),
                'color': color,
                'radius': self.rng.random() * 2 + 1
            })

    def _check_termination(self):
        if self.game_over:
            self.reward_this_step = -100.0
            return True
        if self.victory:
            self.reward_this_step = 100.0
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.reward_this_step = -50.0 # Penalty for running out of time
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_path()
        self._render_build_sites()
        
        for tower in self.towers:
            self._render_tower(tower)

        for enemy in self.enemies:
            self._render_enemy(enemy)
            
        for proj in self.projectiles:
            self._render_projectile(proj)
            
        for p in self.particles:
            self._render_particle(p)
            
        self._render_cursor()
        
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.screen_flash_timer / 5.0))
            flash_surface.fill((255, 0, 0, alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_path(self):
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, 30)
        pygame.gfxdraw.filled_circle(self.screen, self.WIDTH, 200, 20, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, self.WIDTH, 200, 20, self.COLOR_BASE)

    def _render_build_sites(self):
        for i, site in enumerate(self.build_sites):
            is_occupied = any(t['pos'] == site for t in self.towers)
            if is_occupied:
                continue
            color = (255, 255, 255, 15)
            pygame.draw.rect(self.screen, color, (site[0] - 12, site[1] - 12, 24, 24), 1, border_radius=3)
    
    def _render_cursor(self):
        pos = self.build_sites[self.cursor_index]
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        is_occupied = any(t['pos'] == pos for t in self.towers)
        can_afford = self.resources >= tower_def['cost']

        # Draw range indicator
        range_color = (255, 255, 255, 30)
        if is_occupied or not can_afford:
            range_color = (255, 0, 0, 30)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], tower_def['range'], range_color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], tower_def['range'], (255,255,255,50))

        # Draw cursor box
        cursor_color = self.COLOR_CURSOR
        if is_occupied or not can_afford:
            cursor_color = (255, 80, 80)
        pygame.draw.rect(self.screen, cursor_color, (pos[0] - 14, pos[1] - 14, 28, 28), 2, border_radius=4)

    def _render_tower(self, tower):
        pos = tower['pos']
        tower_def = self.TOWER_TYPES[tower['type']]
        color = tower_def['color']
        
        # Base
        pygame.draw.rect(self.screen, (30,30,40), (pos[0]-10, pos[1]-10, 20, 20), border_radius=3)
        # Turret
        pygame.draw.rect(self.screen, color, (pos[0]-7, pos[1]-7, 14, 14), border_radius=2)
        
    def _render_enemy(self, enemy):
        pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
        radius = 8
        # Health bar
        if enemy['health'] < enemy['max_health']:
            bar_w = 20
            bar_h = 4
            bar_x = pos[0] - bar_w // 2
            bar_y = pos[1] - radius - bar_h - 2
            health_pct = enemy['health'] / enemy['max_health']
            pygame.draw.rect(self.screen, (80, 0, 0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0, 200, 0), (bar_x, bar_y, int(bar_w * health_pct), bar_h))
            
        # Body with outline
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_BG)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius-1, self.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_BG)

    def _render_projectile(self, proj):
        pos = (int(proj['pos'][0]), int(proj['pos'][1]))
        color = self.TOWER_TYPES[proj['type']]['color']
        radius = 4 if proj['type'] == 0 else 6
        
        # Glow
        glow_color = color + (80,)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 2, glow_color)
        
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
        
    def _render_particle(self, p):
        life_pct = max(0, p['lifetime'] / 20.0)
        radius = int(p['radius'] * life_pct)
        if radius < 1: return
        color = p['color']
        pos = (int(p['pos'][0]), int(p['pos'][1]))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
        
    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, shadow_color):
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Wave Number
        wave_text = f"Wave: {self.wave_number}/{self.MAX_WAVES}"
        draw_text(wave_text, self.FONT_UI, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)
        
        # Resources
        res_text = f"Resources: ${self.resources}"
        draw_text(res_text, self.FONT_UI, self.COLOR_TEXT, (10, self.HEIGHT - 30), self.COLOR_TEXT_SHADOW)
        
        # Base Health
        health_text = "Base Health"
        text_w = self.FONT_UI.size(health_text)[0]
        draw_text(health_text, self.FONT_UI, self.COLOR_TEXT, (self.WIDTH - text_w - 10, 10), self.COLOR_TEXT_SHADOW)
        
        bar_w, bar_h = 150, 15
        bar_x, bar_y = self.WIDTH - bar_w - 10, 35
        health_pct = self.base_health / 100.0
        pygame.draw.rect(self.screen, (80, 0, 0), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_BASE, (bar_x, bar_y, int(bar_w * health_pct), bar_h), border_radius=3)

        # Selected Tower Info (near cursor)
        cursor_pos = self.build_sites[self.cursor_index]
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        info_text = f"{tower_def['name']} (${tower_def['cost']})"
        info_pos = (cursor_pos[0] + 20, cursor_pos[1] - 30)
        draw_text(info_text, self.FONT_POPUP, self.COLOR_TEXT, info_pos, self.COLOR_TEXT_SHADOW)

        # Game Over / Victory
        if self.game_over or self.victory:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if self.victory else "GAME OVER"
            color = (100, 255, 100) if self.victory else (255, 100, 100)
            text_w, text_h = self.FONT_GAMEOVER.size(message)
            draw_text(message, self.FONT_GAMEOVER, color, (self.WIDTH/2 - text_w/2, self.HEIGHT/2 - text_h/2), self.COLOR_TEXT_SHADOW)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "resources": self.resources,
            "enemies": len(self.enemies)
        }

# Example of how to run the environment
if __name__ == '__main__':
    # The main block now requires a visible display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    running = True
    
    total_reward = 0
    
    while running:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        clock.tick(30)
        
    pygame.quit()