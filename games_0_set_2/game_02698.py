
# Generated: 2025-08-27T21:10:01.169147
# Source Brief: brief_02698.md
# Brief Index: 2698

        
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
        "Controls: Use arrow keys to select a tower plot. Press Space to build the selected tower. Press Shift to cycle between tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down tower defense game. Place towers to defend your base from waves of enemies. Survive 10 waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 30 * 180 # 3 minutes at 30fps
    WIN_WAVE = 10

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_PATH = (40, 50, 80)
    COLOR_PLOT = (60, 70, 100)
    COLOR_PLOT_HOVER = (255, 255, 0)
    COLOR_BASE = (0, 150, 200)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_TEXT = (230, 230, 230)
    TOWER_COLORS = [(255, 200, 0), (0, 255, 150)]
    PROJECTILE_COLORS = [(255, 255, 0), (100, 255, 200)]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- Game Data ---
        self._define_game_data()

        self.np_random = None
        self.reset()
        
        self.validate_implementation()


    def _define_game_data(self):
        """Set up static game data like paths and tower stats."""
        self.path = [
            (100, -20), (100, 150), (320, 150),
            (320, 300), (540, 300), (540, 420)
        ]
        self.tower_plots = [
            (210, 80), (210, 220), (430, 230), (430, 370),
            (150, 250), (50, 300)
        ]
        self.base_rect = pygame.Rect(self.SCREEN_WIDTH - 100, self.SCREEN_HEIGHT, 100, 20)
        self.tower_stats = [
            {"name": "Gatling", "cost": 50, "range": 80, "damage": 2, "fire_rate": 5},
            {"name": "Cannon", "cost": 120, "range": 120, "damage": 15, "fire_rate": 45}
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            random.seed(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = 100
        self.resources = 150
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.current_wave = 0
        self.wave_timer = 150  # Time until the first wave
        self.wave_in_progress = False
        self.enemies_to_spawn = []

        self.cursor_index = 0
        self.selected_tower_type = 0
        self.last_space_press = False
        self.last_shift_press = False
        self.base_flash_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self._handle_input(movement, space_held, shift_held)
        
        if not self.game_over:
            reward += self._update_game_state()
        
        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            if self.win:
                reward += 100
            else:
                reward += -100 # Lost due to health or timeout
            self.score += reward
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        """Process player actions."""
        # --- Cursor Movement ---
        if movement != 0:
            # This is a bit of a hack to make discrete arrow presses move the cursor
            # A better RL agent would learn to pulse the action.
            # For human play, this feels okay.
            if movement == 1: # Up
                self.cursor_index = (self.cursor_index - 2) % len(self.tower_plots)
            elif movement == 2: # Down
                self.cursor_index = (self.cursor_index + 2) % len(self.tower_plots)
            elif movement == 3: # Left
                self.cursor_index = (self.cursor_index - 1) % len(self.tower_plots)
            elif movement == 4: # Right
                self.cursor_index = (self.cursor_index + 1) % len(self.tower_plots)

        # --- Cycle Tower Type ---
        if shift_held and not self.last_shift_press:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_stats)
        self.last_shift_press = shift_held

        # --- Place Tower ---
        if space_held and not self.last_space_press:
            plot_pos = self.tower_plots[self.cursor_index]
            stats = self.tower_stats[self.selected_tower_type]
            
            can_afford = self.resources >= stats["cost"]
            is_occupied = any(t['pos'] == plot_pos for t in self.towers)

            if can_afford and not is_occupied:
                self.resources -= stats["cost"]
                self.towers.append({
                    "pos": plot_pos,
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                    "flash": 0
                })
                # sfx: build_tower.wav
                self._create_particles(plot_pos, 20, self.TOWER_COLORS[self.selected_tower_type], 2, 4)

        self.last_space_press = space_held

    def _update_game_state(self):
        """Advance all game logic by one frame."""
        reward = 0
        
        self._update_wave_manager()
        reward += self._update_towers()
        self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()

        if self.base_flash_timer > 0:
            self.base_flash_timer -= 1
            
        return reward

    def _update_wave_manager(self):
        if not self.wave_in_progress and not self.win:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                if self.current_wave > self.WIN_WAVE:
                    self.win = True
                    return
                
                self.wave_in_progress = True
                num_enemies = 5 + self.current_wave * 2
                enemy_health = 10 * (1.05 ** (self.current_wave - 1))
                enemy_speed = 1.0 + self.current_wave * 0.05
                self.enemies_to_spawn = [
                    {"health": enemy_health, "speed": enemy_speed}
                    for _ in range(num_enemies)
                ]

        if self.wave_in_progress and self.enemies_to_spawn:
            if self.steps % 20 == 0: # Spawn interval
                spawn_data = self.enemies_to_spawn.pop(0)
                self.enemies.append({
                    "pos": pygame.Vector2(self.path[0]),
                    "max_health": spawn_data["health"],
                    "health": spawn_data["health"],
                    "speed": spawn_data["speed"],
                    "path_index": 1,
                    "value": 5 + self.current_wave
                })

        if self.wave_in_progress and not self.enemies and not self.enemies_to_spawn:
            self.wave_in_progress = False
            self.wave_timer = 300 # Time between waves
            self.resources += 50 + self.current_wave * 10 # End of wave bonus

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            if tower['flash'] > 0:
                tower['flash'] -= 1
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            
            if tower['cooldown'] == 0:
                stats = self.tower_stats[tower['type']]
                target = self._find_target(tower['pos'], stats['range'])
                if target:
                    # sfx: fire_weapon.wav
                    tower['cooldown'] = stats['fire_rate']
                    tower['flash'] = 3
                    self.projectiles.append({
                        "pos": pygame.Vector2(tower['pos']),
                        "target": target,
                        "type": tower['type'],
                        "speed": 8
                    })
        return reward

    def _find_target(self, pos, range_):
        in_range_enemies = []
        for enemy in self.enemies:
            dist = pygame.Vector2(pos).distance_to(enemy['pos'])
            if dist <= range_:
                in_range_enemies.append(enemy)
        
        if not in_range_enemies:
            return None
        
        # Target enemy closest to the base (farthest along the path)
        return max(in_range_enemies, key=lambda e: e['path_index'] + pygame.Vector2(self.path[e['path_index']-1]).distance_to(e['pos']))

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            target_pos = proj['target']['pos']
            direction = (target_pos - proj['pos']).normalize()
            proj['pos'] += direction * proj['speed']
            
            if proj['pos'].distance_to(target_pos) < 5:
                # Projectile hits handled in _update_enemies to avoid double-counting
                # when an enemy is destroyed. We mark it for hit-check there.
                if 'hit_by' not in proj['target']:
                    proj['target']['hit_by'] = []
                proj['target']['hit_by'].append(proj)
                self.projectiles.remove(proj)

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            # Process projectile hits first
            if 'hit_by' in enemy:
                for proj in enemy['hit_by']:
                    stats = self.tower_stats[proj['type']]
                    enemy['health'] -= stats['damage']
                    reward += 0.1 # Reward for hitting
                    # sfx: hit_enemy.wav
                    self._create_particles(enemy['pos'], 5, self.PROJECTILE_COLORS[proj['type']], 1, 2)
                del enemy['hit_by']

            if enemy['health'] <= 0:
                # sfx: enemy_destroyed.wav
                reward += 1.0 # Reward for kill
                self.score += 10 # Add to visual score display
                self.resources += enemy['value']
                self._create_particles(enemy['pos'], 30, (255, 150, 50), 2, 5, 0.95)
                self.enemies.remove(enemy)
                continue

            # Movement
            target_waypoint = pygame.Vector2(self.path[enemy['path_index']])
            direction = (target_waypoint - enemy['pos'])
            
            if direction.length() < enemy['speed']:
                enemy['pos'] = target_waypoint
                enemy['path_index'] += 1
                if enemy['path_index'] >= len(self.path):
                    # sfx: base_hit.wav
                    self.base_health -= 10
                    self.base_flash_timer = 10
                    reward -= 1.0 # Penalty for reaching base
                    self.enemies.remove(enemy)
            else:
                enemy['pos'] += direction.normalize() * enemy['speed']
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['radius'] *= p.get('decay', 0.9)
            p['lifespan'] -= 1
            if p['lifespan'] <= 0 or p['radius'] < 0.5:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            return True
        if self.win and not self.enemies:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 30)

        # Draw Tower Plots
        for i, pos in enumerate(self.tower_plots):
            color = self.COLOR_PLOT_HOVER if i == self.cursor_index else self.COLOR_PLOT
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 15, color)
        
        # Draw Base Health Flash
        if self.base_flash_timer > 0:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.base_flash_timer / 10))
            s.fill((255, 0, 0, alpha))
            self.screen.blit(s, (0,0))

        # Draw Towers
        for tower in self.towers:
            pos = (int(tower['pos'][0]), int(tower['pos'][1]))
            stats = self.tower_stats[tower['type']]
            color = self.TOWER_COLORS[tower['type']]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, color)
            if tower['flash'] > 0:
                flash_color = (255, 255, 255)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 14, flash_color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 14, flash_color)

        # Draw Projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            color = self.PROJECTILE_COLORS[proj['type']]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, color)

        # Draw Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            size = 8
            rect = pygame.Rect(pos[0] - size, pos[1] - size, size*2, size*2)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect)
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            pygame.draw.rect(self.screen, (50, 50, 50), (rect.left, rect.top - 5, rect.width, 3))
            pygame.draw.rect(self.screen, (100, 255, 100), (rect.left, rect.top - 5, int(rect.width * health_pct), 3))

        # Draw Particles
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p['color'])

    def _render_ui(self):
        # --- Top Bar ---
        ui_bar = pygame.Rect(0, 0, self.SCREEN_WIDTH, 30)
        pygame.draw.rect(self.screen, (10, 15, 25), ui_bar)
        
        # Base Health
        health_text = self.font_small.render(f"Base Health: {max(0, self.base_health)}/100", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 7))
        
        # Resources
        resource_text = self.font_small.render(f"Resources: ${self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(resource_text, (200, 7))

        # Wave
        wave_text = self.font_small.render(f"Wave: {self.current_wave}/{self.WIN_WAVE}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (350, 7))
        
        # --- Tower Selection UI ---
        stats = self.tower_stats[self.selected_tower_type]
        name_text = self.font_small.render(f"Selected: {stats['name']}", True, self.COLOR_TEXT)
        cost_text = self.font_small.render(f"Cost: ${stats['cost']}", True, self.COLOR_TEXT)
        self.screen.blit(name_text, (470, 7))
        self.screen.blit(cost_text, (470, 35))
        
        # --- Cursor ---
        cursor_pos = self.tower_plots[self.cursor_index]
        pulse = abs(math.sin(self.steps * 0.1))
        radius = int(18 + pulse * 4)
        alpha = int(100 + pulse * 100)
        
        s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        color = self.TOWER_COLORS[self.selected_tower_type]
        pygame.gfxdraw.aacircle(s, radius, radius, radius-1, (color[0], color[1], color[2], alpha))
        pygame.gfxdraw.aacircle(s, radius, radius, radius-2, (color[0], color[1], color[2], alpha))
        self.screen.blit(s, (cursor_pos[0] - radius, cursor_pos[1] - radius))

        # --- Game Over Screen ---
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0,0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "health": self.base_health,
            "resources": self.resources,
            "towers": len(self.towers),
            "enemies": len(self.enemies)
        }
        
    def _create_particles(self, pos, count, color, min_speed, max_speed, decay=0.9):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "radius": self.np_random.uniform(2, 5),
                "color": color,
                "lifespan": self.np_random.integers(15, 30),
                "decay": decay
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

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'mac' etc. depending on your OS

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Human Input to Action ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()