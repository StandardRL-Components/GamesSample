
# Generated: 2025-08-27T14:26:17.490898
# Source Brief: brief_00682.md
# Brief Index: 682

        
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

    user_guide = (
        "Controls: Use arrow keys to select a tower plot. Press Shift to cycle tower types. Press Space to build."
    )

    game_description = (
        "A minimalist tower defense game. Place towers to defend your base from 10 waves of enemies."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # Game Constants
        self.MAX_STEPS = 2000 # Increased for better playability
        self.MAX_WAVES = 10
        self.INITIAL_RESOURCES = 100
        self.BASE_POS = (self.WIDTH - 40, self.HEIGHT - 40)
        
        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_PLOT = (60, 70, 80)
        self.COLOR_PLOT_SELECTED = (255, 255, 0)
        self.COLOR_BASE = (0, 150, 50)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_PARTICLE = (255, 200, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.TOWER_COLORS = [(80, 120, 220), (180, 80, 220)]

        # Game Data
        self._define_level()
        self._define_towers()
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def _define_level(self):
        self.path_waypoints = [
            (-20, 50), (150, 50), (150, 250), (450, 250), (450, 150),
            (self.WIDTH-40, 150), self.BASE_POS
        ]
        self.tower_plots = [
            (80, 120), (220, 120), (220, 200), (220, 320),
            (380, 200), (380, 320), (520, 200), (520, 80)
        ]
    
    def _define_towers(self):
        self.tower_types = [
            {
                "name": "Machine Gun", "cost": 30, "range": 80, 
                "damage": 2, "fire_rate": 5, "color": self.TOWER_COLORS[0]
            },
            {
                "name": "Cannon", "cost": 75, "range": 120, 
                "damage": 15, "fire_rate": 45, "color": self.TOWER_COLORS[1]
            }
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.resources = self.INITIAL_RESOURCES
        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_cooldown = 90 # 3 seconds at 30fps

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.enemies_to_spawn = []
        
        self.cursor_index = 0
        self.selected_tower_type_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.step_reward = -0.01 # Living penalty

        self._handle_input(action)
        
        if not self.game_over:
            self._update_waves()
            self._update_towers()
            self._update_enemies()
            self._update_projectiles()
        
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if self.game_won:
            self.step_reward += 100
        
        if self.auto_advance:
            self.clock.tick(30)

        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        if movement > 0:
            # A simple mapping to navigate the 1D list of plots
            if movement == 1: self.cursor_index = (self.cursor_index - 2) % len(self.tower_plots) # Up
            if movement == 2: self.cursor_index = (self.cursor_index + 2) % len(self.tower_plots) # Down
            if movement == 3: self.cursor_index = (self.cursor_index - 1) % len(self.tower_plots) # Left
            if movement == 4: self.cursor_index = (self.cursor_index + 1) % len(self.tower_plots) # Right

        # --- Cycle Tower Type (on press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.tower_types)
        
        # --- Place Tower (on press) ---
        if space_held and not self.prev_space_held:
            plot_pos = self.tower_plots[self.cursor_index]
            tower_spec = self.tower_types[self.selected_tower_type_idx]
            
            can_afford = self.resources >= tower_spec["cost"]
            is_occupied = any(t['pos'] == plot_pos for t in self.towers)

            if can_afford and not is_occupied:
                self.resources -= tower_spec["cost"]
                self.towers.append({
                    "pos": plot_pos,
                    "type": self.selected_tower_type_idx,
                    "cooldown": 0,
                })
                # sfx: build_tower.wav

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_waves(self):
        if self.game_won: return

        if not self.wave_in_progress:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0 and self.wave_number < self.MAX_WAVES:
                self.wave_number += 1
                self.wave_in_progress = True
                self._prepare_wave()
        else:
            if self.enemies_to_spawn:
                self.spawn_timer -= 1
                if self.spawn_timer <= 0:
                    self.enemies.append(self.enemies_to_spawn.pop(0))
                    self.spawn_timer = 15 # Spawn interval
            elif not self.enemies:
                self.wave_in_progress = False
                self.wave_cooldown = 150 # 5 seconds between waves
                if self.wave_number >= self.MAX_WAVES:
                    self.game_won = True

    def _prepare_wave(self):
        num_enemies = 10 + (self.wave_number - 1) * 2
        base_health = 10 + (self.wave_number - 1) * 5
        base_speed = 0.8 + (self.wave_number - 1) * 0.05
        
        for _ in range(num_enemies):
            health = int(base_health * random.uniform(0.9, 1.1))
            speed = base_speed * random.uniform(0.9, 1.1)
            self.enemies_to_spawn.append({
                "pos": list(self.path_waypoints[0]),
                "health": health,
                "max_health": health,
                "speed": speed,
                "target_wp_idx": 1
            })
        self.spawn_timer = 0
    
    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            spec = self.tower_types[tower['type']]
            target = None
            min_dist = spec['range']

            for enemy in self.enemies:
                dist = math.hypot(enemy['pos'][0] - tower['pos'][0], enemy['pos'][1] - tower['pos'][1])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                tower['cooldown'] = spec['fire_rate']
                self.projectiles.append({
                    "start_pos": list(tower['pos']),
                    "target_pos": list(target['pos']),
                    "pos": list(tower['pos']),
                    "damage": spec['damage'],
                    "speed": 8,
                })
                # sfx: shoot.wav

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            target_pos = self.path_waypoints[enemy['target_wp_idx']]
            dx = target_pos[0] - enemy['pos'][0]
            dy = target_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < enemy['speed']:
                enemy['target_wp_idx'] += 1
                if enemy['target_wp_idx'] >= len(self.path_waypoints):
                    self.enemies.remove(enemy)
                    self.game_over = True
                    self.step_reward -= 10
                    # sfx: base_damage.wav
                    continue
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']
    
    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            target_enemy_hit = None
            for enemy in self.enemies:
                radius = max(5, int(enemy['health'] / enemy['max_health'] * 12))
                if math.hypot(proj['pos'][0] - enemy['pos'][0], proj['pos'][1] - enemy['pos'][1]) < radius:
                    target_enemy_hit = enemy
                    break
            
            if target_enemy_hit:
                target_enemy_hit['health'] -= proj['damage']
                self.step_reward += 0.1
                # sfx: hit.wav
                if target_enemy_hit['health'] <= 0:
                    self.score += 10
                    self.resources += 5
                    self.step_reward += 1
                    for _ in range(15):
                        self.particles.append(self._create_particle(target_enemy_hit['pos']))
                    self.enemies.remove(target_enemy_hit)
                    # sfx: enemy_destroyed.wav
                self.projectiles.remove(proj)
            else:
                # Move projectile towards its original target position
                dx = proj['target_pos'][0] - proj['pos'][0]
                dy = proj['target_pos'][1] - proj['pos'][1]
                dist = math.hypot(dx, dy)
                if dist < proj['speed']:
                    self.projectiles.remove(proj)
                else:
                    proj['pos'][0] += (dx / dist) * proj['speed']
                    proj['pos'][1] += (dy / dist) * proj['speed']

    def _create_particle(self, pos):
        return {
            "pos": list(pos),
            "vel": [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)],
            "life": 20, "size": random.randint(2, 4)
        }

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.game_over or self.game_won or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_path()
        self._render_placement_spots_and_cursor()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "enemies_left": len(self.enemies) + len(self.enemies_to_spawn)
        }

    def _render_path(self):
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, 40)

    def _render_placement_spots_and_cursor(self):
        for i, pos in enumerate(self.tower_plots):
            color = self.COLOR_PLOT_SELECTED if i == self.cursor_index else self.COLOR_PLOT
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 15, color)
            if i == self.cursor_index:
                 pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 15, (*color, 50))


    def _render_base(self):
        base_rect = pygame.Rect(self.BASE_POS[0]-20, self.BASE_POS[1]-20, 40, 40)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

    def _render_towers(self):
        for tower in self.towers:
            spec = self.tower_types[tower['type']]
            pos = tower['pos']
            color = spec['color']
            
            # Draw range indicator
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), spec['range'], (*color, 50))

            # Draw tower triangle
            p1 = (pos[0], pos[1] - 10)
            p2 = (pos[0] - 8, pos[1] + 6)
            p3 = (pos[0] + 8, pos[1] + 6)
            pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), color)
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), color)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = enemy['pos']
            health_ratio = enemy['health'] / enemy['max_health']
            radius = max(4, int(12 * health_ratio))
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_ENEMY)

    def _render_projectiles(self):
        for proj in self.projectiles:
            pygame.draw.aaline(self.screen, self.COLOR_PROJECTILE, proj['start_pos'], proj['pos'], 2)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*self.COLOR_PARTICLE, alpha)
            rect = pygame.Rect(int(p['pos'][0]), int(p['pos'][1]), p['size'], p['size'])
            pygame.draw.rect(self.screen, color, rect)

    def _render_ui(self):
        # Wave
        wave_text = self.font_m.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        # Score
        score_text = self.font_m.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Resources
        res_text = self.font_m.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (10, self.HEIGHT - res_text.get_height() - 10))

        # Selected Tower
        tower_spec = self.tower_types[self.selected_tower_type_idx]
        tower_name = self.font_s.render(f"Build: {tower_spec['name']}", True, self.COLOR_TEXT)
        tower_cost = self.font_s.render(f"Cost: {tower_spec['cost']}", True, self.COLOR_TEXT)
        self.screen.blit(tower_name, (10, self.HEIGHT - res_text.get_height() - tower_name.get_height() - 15))
        self.screen.blit(tower_cost, (10, self.HEIGHT - res_text.get_height() - tower_cost.get_height() * 2 - 15))

        # Game Over / Win Message
        if self.game_over:
            msg = self.font_l.render("GAME OVER", True, self.COLOR_ENEMY)
            self.screen.blit(msg, (self.WIDTH//2 - msg.get_width()//2, self.HEIGHT//2 - msg.get_height()//2))
        elif self.game_won:
            msg = self.font_l.render("YOU WIN!", True, self.COLOR_PLOT_SELECTED)
            self.screen.blit(msg, (self.WIDTH//2 - msg.get_width()//2, self.HEIGHT//2 - msg.get_height()//2))

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
    # This block allows you to run the game directly for testing
    # It requires pygame to be fully installed with display support
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'mac', etc.
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    done = False
    total_reward = 0
    
    # Map keyboard keys to MultiDiscrete actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while not done:
        # --- Human Input ---
        movement_action = 0
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        for key, move_val in key_to_action.items():
            if keys[key]:
                movement_action = move_val
                break # Only one movement at a time
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already the rendered frame
        # We just need to show it on the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Print info
        print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Resources: {info['resources']}")

    print("Game Over!")
    print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    pygame.quit()