
# Generated: 2025-08-27T16:34:13.570016
# Source Brief: brief_01262.md
# Brief Index: 1262

        
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
        "Controls: Arrows to select a tower zone. Space to build a tower. Shift to upgrade a tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of zombies. Place and upgrade towers to survive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_PATH = (60, 60, 80)
    COLOR_ZONE = (45, 50, 70)
    COLOR_ZONE_SELECTED = (200, 200, 100)
    COLOR_BASE = (0, 150, 136)
    COLOR_BASE_GLOW = (0, 150, 136, 50)
    
    COLOR_TOWER = (0, 188, 212)
    COLOR_TOWER_UPGRADED = (255, 193, 7)
    
    COLOR_ZOMBIE = (244, 67, 54)
    COLOR_ZOMBIE_HEALTH_BG = (80, 20, 20)
    
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_TEXT = (230, 230, 230)

    # Game Parameters
    MAX_STEPS = 3000
    TOTAL_WAVES = 10
    
    INITIAL_RESOURCES = 150
    TOWER_COST = 75
    UPGRADE_COST = 100
    ZOMBIE_KILL_REWARD = 15
    
    ZOMBIE_BASE_HEALTH = 100
    ZOMBIE_HEALTH_WAVE_SCALER = 1.1
    ZOMBIE_SPEED = 0.8
    
    TOWER_RANGE = 90
    TOWER_COOLDOWN = 45 # frames
    TOWER_DAMAGE = 50
    TOWER_UPGRADE_DMG_BONUS = 75
    
    ZOMBIE_SPAWN_COOLDOWN = 10 # frames between zombies in a wave

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 52)
        
        self._define_path_and_zones()
        self.reset()
        self.validate_implementation()
        
    def _define_path_and_zones(self):
        self.path_points = [
            (-20, 100), (120, 100), (120, 300), 
            (480, 300), (480, 150), (self.WIDTH + 20, 150)
        ]
        self.path_segments = []
        for i in range(len(self.path_points) - 1):
            p1 = self.path_points[i]
            p2 = self.path_points[i+1]
            length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            self.path_segments.append({'p1': p1, 'p2': p2, 'length': length})
        
        self.total_path_length = sum(s['length'] for s in self.path_segments)

        self.tower_zones = [
            pygame.Rect(60, 150, 50, 50),
            pygame.Rect(180, 240, 50, 50),
            pygame.Rect(420, 200, 50, 50),
            pygame.Rect(540, 100, 50, 50)
        ]
        self.base_rect = pygame.Rect(self.WIDTH - 20, 0, 20, self.HEIGHT)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.final_message = ""
        
        self.resources = self.INITIAL_RESOURCES
        self.current_wave = 0
        self.wave_cleared = True
        self.wave_countdown = 90 # frames before first wave
        
        self.zombies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.zombies_to_spawn = 0
        self.zombie_spawn_timer = 0
        
        self.selected_zone_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        reward = -0.001 # Small time penalty to encourage speed
        
        if not self.game_over:
            self._handle_input(action)
            
            kill_reward = self._update_game_logic()
            reward += kill_reward
            
            wave_reward = self._update_waves()
            reward += wave_reward

        terminated = self._check_termination()
        if terminated and not self.victory:
             reward = -100.0
        elif terminated and self.victory:
             reward = 100.0

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement in [1, 2, 3, 4]: # Arrow keys
            # A simple mapping: up/left decrease index, down/right increase
            if movement in [1, 3]: # Up, Left
                self.selected_zone_idx = (self.selected_zone_idx - 1) % len(self.tower_zones)
            else: # Down, Right
                self.selected_zone_idx = (self.selected_zone_idx + 1) % len(self.tower_zones)

        if space_held and not self.prev_space_held:
            self._place_tower(self.selected_zone_idx)
        
        if shift_held and not self.prev_shift_held:
            self._upgrade_tower(self.selected_zone_idx)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_game_logic(self):
        self._update_towers()
        kill_reward = self._update_projectiles()
        self._update_zombies()
        self._update_particles()
        return kill_reward

    def _update_waves(self):
        reward = 0
        if self.wave_cleared and self.current_wave <= self.TOTAL_WAVES:
            self.wave_countdown -= 1
            if self.wave_countdown <= 0:
                self.current_wave += 1
                if self.current_wave > self.TOTAL_WAVES:
                    self.victory = True
                    return reward

                self.wave_cleared = False
                self.zombies_to_spawn = 8 + 2 * self.current_wave
                self.zombie_spawn_timer = 0
                reward = 10.0 # Wave start bonus
        
        if not self.wave_cleared and self.zombies_to_spawn > 0:
            self.zombie_spawn_timer -= 1
            if self.zombie_spawn_timer <= 0:
                self._spawn_zombie()
                self.zombies_to_spawn -= 1
                self.zombie_spawn_timer = self.ZOMBIE_SPAWN_COOLDOWN
        
        if not self.wave_cleared and not self.zombies and self.zombies_to_spawn == 0:
            self.wave_cleared = True
            self.wave_countdown = 150 # Time until next wave
            
        return reward

    def _spawn_zombie(self):
        health = self.ZOMBIE_BASE_HEALTH * (self.ZOMBIE_HEALTH_WAVE_SCALER ** (self.current_wave - 1))
        zombie = {
            'pos': list(self.path_points[0]),
            'dist_traveled': 0,
            'max_health': health,
            'health': health,
            'size': 14,
        }
        self.zombies.append(zombie)

    def _update_zombies(self):
        for z in list(self.zombies):
            z['dist_traveled'] += self.ZOMBIE_SPEED
            
            current_dist = 0
            for i, seg in enumerate(self.path_segments):
                if current_dist + seg['length'] >= z['dist_traveled']:
                    # This is the segment the zombie is on
                    ratio = (z['dist_traveled'] - current_dist) / seg['length']
                    p1 = seg['p1']
                    p2 = seg['p2']
                    z['pos'][0] = p1[0] + ratio * (p2[0] - p1[0])
                    z['pos'][1] = p1[1] + ratio * (p2[1] - p1[1])
                    break
                current_dist += seg['length']
            
            if z['dist_traveled'] >= self.total_path_length:
                self.zombies.remove(z)
                self.game_over = True
                # sfx: base_alarm.wav

    def _update_towers(self):
        for t in self.towers:
            t['cooldown'] = max(0, t['cooldown'] - 1)
            if t['cooldown'] == 0:
                # Find target: furthest zombie in range
                target = None
                max_dist = -1
                for z in self.zombies:
                    dist = math.hypot(t['pos'][0] - z['pos'][0], t['pos'][1] - z['pos'][1])
                    if dist <= t['range'] and z['dist_traveled'] > max_dist:
                        max_dist = z['dist_traveled']
                        target = z
                
                if target:
                    self._fire_projectile(t, target)
                    t['cooldown'] = self.TOWER_COOLDOWN

    def _fire_projectile(self, tower, target):
        proj = {
            'pos': list(tower['pos']),
            'target': target,
            'damage': tower['damage'],
            'speed': 5,
        }
        self.projectiles.append(proj)
        # sfx: laser_shoot.wav

    def _update_projectiles(self):
        kill_reward = 0
        for p in list(self.projectiles):
            if p['target'] not in self.zombies:
                self.projectiles.remove(p)
                continue

            dx = p['target']['pos'][0] - p['pos'][0]
            dy = p['target']['pos'][1] - p['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < p['speed']:
                p['target']['health'] -= p['damage']
                self._create_hit_particles(p['pos'])
                self.projectiles.remove(p)
                # sfx: projectile_hit.wav
                
                if p['target']['health'] <= 0:
                    self._create_death_particles(p['target']['pos'])
                    self.zombies.remove(p['target'])
                    self.resources += self.ZOMBIE_KILL_REWARD
                    self.score += 10
                    kill_reward += 0.1
                    # sfx: zombie_die.wav
            else:
                p['pos'][0] += (dx / dist) * p['speed']
                p['pos'][1] += (dy / dist) * p['speed']
        return kill_reward

    def _update_particles(self):
        for p in list(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
    
    def _create_hit_particles(self, pos):
        for _ in range(5):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(5, 10),
                'color': self.COLOR_PROJECTILE,
                'size': random.randint(1, 2)
            })

    def _create_death_particles(self, pos):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(10, 20),
                'color': self.COLOR_ZOMBIE,
                'size': random.randint(2, 4)
            })
            
    def _place_tower(self, zone_idx):
        if self.resources >= self.TOWER_COST:
            zone_rect = self.tower_zones[zone_idx]
            is_occupied = any(t['zone_idx'] == zone_idx for t in self.towers)
            
            if not is_occupied:
                self.resources -= self.TOWER_COST
                tower = {
                    'pos': zone_rect.center,
                    'zone_idx': zone_idx,
                    'level': 1,
                    'range': self.TOWER_RANGE,
                    'cooldown': 0,
                    'damage': self.TOWER_DAMAGE,
                }
                self.towers.append(tower)
                # sfx: build_tower.wav
    
    def _upgrade_tower(self, zone_idx):
        if self.resources >= self.UPGRADE_COST:
            for t in self.towers:
                if t['zone_idx'] == zone_idx and t['level'] == 1:
                    self.resources -= self.UPGRADE_COST
                    t['level'] = 2
                    t['damage'] += self.TOWER_UPGRADE_DMG_BONUS
                    # sfx: upgrade_tower.wav
                    break
                    
    def _check_termination(self):
        if self.victory:
            self.game_over = True
            self.final_message = "VICTORY!"
        if self.game_over and not self.victory:
            self.final_message = "GAME OVER"
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            if not self.final_message:
                self.final_message = "TIME LIMIT REACHED"
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_points, 10)
        
        # Draw base
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.base_rect)
        glow_surf = pygame.Surface((self.base_rect.width * 2, self.base_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_BASE_GLOW, glow_surf.get_rect(), border_radius=10)
        self.screen.blit(glow_surf, (self.base_rect.x - self.base_rect.width // 2, self.base_rect.y))
        
        # Draw tower zones
        for i, zone in enumerate(self.tower_zones):
            color = self.COLOR_ZONE_SELECTED if i == self.selected_zone_idx else self.COLOR_ZONE
            pygame.draw.rect(self.screen, color, zone, 2, border_radius=3)

        # Draw towers
        for t in self.towers:
            color = self.COLOR_TOWER_UPGRADED if t['level'] > 1 else self.COLOR_TOWER
            p1 = (t['pos'][0], t['pos'][1] - 12)
            p2 = (t['pos'][0] - 10, t['pos'][1] + 8)
            p3 = (t['pos'][0] + 10, t['pos'][1] + 8)
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), color)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), color)
            # Range indicator
            pygame.gfxdraw.aacircle(self.screen, int(t['pos'][0]), int(t['pos'][1]), int(t['range']), (*color, 50))
            
        # Draw zombies
        for z in self.zombies:
            x, y = int(z['pos'][0]), int(z['pos'][1])
            size = int(z['size'])
            z_rect = pygame.Rect(x - size//2, y - size//2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z_rect)
            # Health bar
            health_ratio = max(0, z['health'] / z['max_health'])
            bar_w = size * 1.5
            bar_bg = pygame.Rect(x - bar_w//2, y - size, bar_w, 4)
            bar_fg = pygame.Rect(x - bar_w//2, y - size, bar_w * health_ratio, 4)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE_HEALTH_BG, bar_bg)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, bar_fg)

        # Draw projectiles
        for p in self.projectiles:
            x, y = int(p['pos'][0]), int(p['pos'][1])
            pygame.gfxdraw.aacircle(self.screen, x, y, 4, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 4, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, x, y, 6, (*self.COLOR_PROJECTILE, 100))
            
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 10))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            # Particles are too small for gfxdraw, simple rect is fine
            pygame.draw.rect(self.screen, color, (pos[0], pos[1], p['size'], p['size']))

    def _render_ui(self):
        # UI Text
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        resources_text = self.font_ui.render(f"RESOURCES: {self.resources}", True, self.COLOR_TEXT)
        wave_text = self.font_ui.render(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(resources_text, (10, 35))
        self.screen.blit(wave_text, (10, 60))
        
        # Wave countdown message
        if self.wave_cleared and self.wave_countdown > 0 and self.current_wave < self.TOTAL_WAVES:
            msg = f"Next wave in {math.ceil(self.wave_countdown / 30)}"
            msg_surf = self.font_msg.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)
            
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg_surf = self.font_msg.render(self.final_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "resources": self.resources,
            "wave": self.current_wave,
            "zombies_left": len(self.zombies) + self.zombies_to_spawn,
        }

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

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Change to 'windows' or 'mac' as needed, or remove for default
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    terminated = False
    clock = pygame.time.Clock()
    
    # --- Key mapping for human play ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    
    action = np.array([0, 0, 0])
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        
        # Reset movement
        action[0] = 0
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Human play runs at 30 FPS
        
    env.close()
    pygame.quit()