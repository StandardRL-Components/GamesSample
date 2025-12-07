
# Generated: 2025-08-28T02:17:00.830410
# Source Brief: brief_01657.md
# Brief Index: 1657

        
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
        "Controls: ↑↓←→ to select a turret spot. Shift to cycle turret types. Space to place a turret."
    )

    game_description = (
        "Defend your base from waves of zombies by strategically placing turrets."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000 * 30 # Approx 30 seconds per wave for 10 waves
    TOTAL_WAVES = 10

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 50)
    COLOR_BASE = (0, 200, 100)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_TURRET_SPOT = (60, 60, 70)
    COLOR_TURRET_SPOT_SELECTED = (255, 255, 0)
    COLOR_ZOMBIE = (220, 50, 50)
    COLOR_HEALTH_BAR_BG = (80, 80, 80)
    COLOR_HEALTH_BAR_FG = (50, 220, 50)
    COLOR_TEXT = (230, 230, 230)
    
    TURRET_TYPES = [
        {"name": "Gun", "cost": 50, "range": 120, "damage": 12, "fire_rate": 1.0, "color": (100, 150, 255), "proj_speed": 8},
        {"name": "Rapid", "cost": 75, "range": 100, "damage": 6, "fire_rate": 3.0, "color": (255, 150, 50), "proj_speed": 10},
    ]

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        self.base_pos = pygame.math.Vector2(80, self.HEIGHT // 2)
        self.turret_spots = [
            pygame.math.Vector2(180, 100), pygame.math.Vector2(320, 100),
            pygame.math.Vector2(180, 300), pygame.math.Vector2(320, 300),
            pygame.math.Vector2(250, 200), pygame.math.Vector2(400, 200)
        ]
        self.occupied_spots = [False] * len(self.turret_spots)
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = 100
        self.max_base_health = 100
        self.resources = 120
        self.current_wave = 0
        self.wave_in_progress = False
        self.wave_cooldown = self.FPS * 5 # 5 seconds between waves
        
        self.zombies = []
        self.turrets = []
        self.projectiles = []
        self.particles = []
        
        self.occupied_spots = [False] * len(self.turret_spots)
        
        self.selected_turret_spot_idx = 0
        self.selected_turret_type_idx = 0
        
        self.was_space_pressed = False
        self.was_shift_pressed = False
        self.base_damage_timer = 0
        
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.001 # Small penalty for existing
        
        self._handle_input(action)
        
        if not self.game_over:
            reward += self._update_game_state()

        self.score += reward
        self.steps += 1
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if self.game_over and self.base_health <= 0:
            reward -= 100
        elif self.game_over and self.current_wave > self.TOTAL_WAVES:
            reward += 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1

        # Cycle turret spots
        if movement in [1, 3]: # Up/Left
            self.selected_turret_spot_idx = (self.selected_turret_spot_idx - 1) % len(self.turret_spots)
        elif movement in [2, 4]: # Down/Right
            self.selected_turret_spot_idx = (self.selected_turret_spot_idx + 1) % len(self.turret_spots)

        # Cycle turret types (on press, not hold)
        if shift_press and not self.was_shift_pressed:
            self.selected_turret_type_idx = (self.selected_turret_type_idx + 1) % len(self.TURRET_TYPES)
        self.was_shift_pressed = shift_press

        # Place turret (on press, not hold)
        if space_press and not self.was_space_pressed:
            self._place_turret()
        self.was_space_pressed = space_press

    def _place_turret(self):
        spot_idx = self.selected_turret_spot_idx
        if self.occupied_spots[spot_idx]:
            return # Spot already occupied

        turret_type = self.TURRET_TYPES[self.selected_turret_type_idx]
        if self.resources >= turret_type["cost"]:
            self.resources -= turret_type["cost"]
            self.occupied_spots[spot_idx] = True
            
            self.turrets.append({
                "pos": self.turret_spots[spot_idx],
                "type_idx": self.selected_turret_type_idx,
                "cooldown": 0,
                "target": None
            })
            # SFX: build_turret
            self._create_particles(self.turret_spots[spot_idx], 20, self.COLOR_BASE, 1, 3, 20)

    def _update_game_state(self):
        reward = 0
        
        # Wave management
        if not self.wave_in_progress and not self.zombies:
            if self.current_wave > self.TOTAL_WAVES:
                self.game_over = True
                return reward
            
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0:
                reward += 1.0 # Wave complete bonus
                self._start_next_wave()
        
        # Update Turrets
        for turret in self.turrets:
            turret['cooldown'] = max(0, turret['cooldown'] - 1)
            turret_info = self.TURRET_TYPES[turret['type_idx']]
            
            if turret['cooldown'] == 0:
                target = self._find_target(turret['pos'], turret_info['range'])
                if target:
                    # SFX: turret_shoot
                    self.projectiles.append({
                        "pos": turret['pos'].copy(),
                        "target": target,
                        "type_idx": turret['type_idx'],
                        "speed": turret_info['proj_speed'],
                        "damage": turret_info['damage']
                    })
                    turret['cooldown'] = self.FPS / turret_info['fire_rate']
                    self._create_particles(turret['pos'], 3, (255, 255, 100), 2, 4, 5, 0.5)

        # Update Projectiles
        for proj in self.projectiles[:]:
            if proj['target'] not in self.zombies:
                self.projectiles.remove(proj)
                continue
            
            direction = (proj['target']['pos'] - proj['pos']).normalize()
            proj['pos'] += direction * proj['speed']
            
            if proj['pos'].distance_to(proj['target']['pos']) < 10:
                # SFX: zombie_hit
                proj['target']['health'] -= proj['damage']
                self._create_particles(proj['pos'], 5, (255, 150, 50), 1, 2, 10)
                self.projectiles.remove(proj)

        # Update Zombies
        for z in self.zombies[:]:
            if z['health'] <= 0:
                # SFX: zombie_die
                self._create_particles(z['pos'], 30, self.COLOR_ZOMBIE, 1, 4, 25)
                self.zombies.remove(z)
                self.resources += 15
                reward += 0.1
                continue

            direction = (self.base_pos - z['pos']).normalize()
            z['pos'] += direction * z['speed']
            
            if z['pos'].distance_to(self.base_pos) < 20:
                # SFX: base_hit
                self.base_health -= 10
                self.base_damage_timer = 10
                self.zombies.remove(z)
                if self.base_health <= 0:
                    self.base_health = 0
                    self.game_over = True

        # Update Particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        if self.base_damage_timer > 0:
            self.base_damage_timer -= 1
            
        if not self.zombies and self.wave_in_progress:
            self.wave_in_progress = False
            self.wave_cooldown = self.FPS * 5 # Start 5 sec countdown to next wave
            if self.current_wave > self.TOTAL_WAVES:
                self.game_over = True

        return reward

    def _find_target(self, pos, range_):
        valid_targets = [z for z in self.zombies if pos.distance_to(z['pos']) <= range_]
        if not valid_targets:
            return None
        # Target zombie closest to the base
        return min(valid_targets, key=lambda z: z['pos'].distance_to(self.base_pos))

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.TOTAL_WAVES:
            return

        self.wave_in_progress = True
        num_zombies = 5 + (self.current_wave - 1) * 2
        base_health = 10 + (self.current_wave - 1) * 2
        base_speed = 0.5 + (self.current_wave - 1) * 0.05
        
        for _ in range(num_zombies):
            spawn_y = self.np_random.uniform(20, self.HEIGHT - 20)
            spawn_x = self.WIDTH + self.np_random.uniform(20, 100)
            
            self.zombies.append({
                "pos": pygame.math.Vector2(spawn_x, spawn_y),
                "health": base_health,
                "max_health": base_health,
                "speed": base_speed
            })
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Draw turret spots and selection
        for i, spot in enumerate(self.turret_spots):
            color = self.COLOR_TURRET_SPOT_SELECTED if i == self.selected_turret_spot_idx else self.COLOR_TURRET_SPOT
            pygame.gfxdraw.filled_circle(self.screen, int(spot.x), int(spot.y), 20, color)
            pygame.gfxdraw.aacircle(self.screen, int(spot.x), int(spot.y), 20, color)
            if i == self.selected_turret_spot_idx:
                turret_info = self.TURRET_TYPES[self.selected_turret_type_idx]
                pygame.gfxdraw.aacircle(self.screen, int(spot.x), int(spot.y), turret_info['range'], (100, 100, 100))

        # Draw base
        base_color = self.COLOR_BASE_DMG if self.base_damage_timer > 0 else self.COLOR_BASE
        base_rect = pygame.Rect(self.base_pos.x - 20, self.base_pos.y - 20, 40, 40)
        pygame.draw.rect(self.screen, base_color, base_rect, border_radius=5)
        self._draw_health_bar(self.base_pos - pygame.math.Vector2(0, 30), self.base_health, self.max_base_health, 50)
        
        # Draw turrets
        for turret in self.turrets:
            turret_info = self.TURRET_TYPES[turret['type_idx']]
            points = [
                turret['pos'] + pygame.math.Vector2(0, -15).rotate(0),
                turret['pos'] + pygame.math.Vector2(-12, 12).rotate(0),
                turret['pos'] + pygame.math.Vector2(12, 12).rotate(0),
            ]
            pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in points], turret_info['color'])
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in points], turret_info['color'])

        # Draw zombies
        for z in self.zombies:
            z_rect = pygame.Rect(z['pos'].x - 8, z['pos'].y - 8, 16, 16)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z_rect, border_radius=3)
            self._draw_health_bar(z['pos'] - pygame.math.Vector2(0, 15), z['health'], z['max_health'], 20)

        # Draw projectiles
        for proj in self.projectiles:
            color = self.TURRET_TYPES[proj['type_idx']]['color']
            pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'].x), int(proj['pos'].y), 4, color)
            pygame.gfxdraw.aacircle(self.screen, int(proj['pos'].x), int(proj['pos'].y), 4, color)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

    def _render_ui(self):
        # Top bar background
        pygame.draw.rect(self.screen, (15, 15, 25), (0, 0, self.WIDTH, 40))
        
        # Resources
        res_text = self.font_small.render(f"GOLD: {self.resources}", True, (255, 223, 0))
        self.screen.blit(res_text, (10, 10))
        
        # Wave
        wave_str = f"WAVE: {min(self.current_wave, self.TOTAL_WAVES)} / {self.TOTAL_WAVES}"
        if not self.wave_in_progress and self.current_wave <= self.TOTAL_WAVES:
            wave_str += f" (Next in {self.wave_cooldown // self.FPS + 1}s)"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH / 2 - wave_text.get_width() / 2, 10))
        
        # Base Health
        base_health_text = self.font_small.render(f"BASE HP: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(base_health_text, (self.WIDTH - base_health_text.get_width() - 120, 10))
        
        # Selected Turret
        turret_info = self.TURRET_TYPES[self.selected_turret_type_idx]
        sel_turret_text = self.font_small.render(f"{turret_info['name']}", True, turret_info['color'])
        self.screen.blit(sel_turret_text, (150, 10))

        # Game Over / Win message
        if self.game_over:
            if self.base_health <= 0:
                msg = "GAME OVER"
                color = self.COLOR_ZOMBIE
            else:
                msg = "YOU WIN!"
                color = self.COLOR_BASE
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((15, 15, 25, 200))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(end_text, text_rect)

    def _draw_health_bar(self, pos, current, max_val, width):
        if current < 0: current = 0
        ratio = current / max_val
        bg_rect = pygame.Rect(pos.x - width/2, pos.y, width, 5)
        fg_rect = pygame.Rect(pos.x - width/2, pos.y, width * ratio, 5)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect, border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, fg_rect, border_radius=2)

    def _create_particles(self, pos, count, color, min_speed, max_speed, max_life, spread=2*math.pi):
        for _ in range(count):
            angle = self.np_random.uniform(0, spread) - spread/2
            speed = self.np_random.uniform(min_speed, max_speed)
            velocity = pygame.math.Vector2(speed, 0).rotate(math.degrees(angle))
            self.particles.append({
                "pos": pos.copy(),
                "vel": velocity,
                "life": self.np_random.integers(max_life // 2, max_life),
                "max_life": max_life,
                "color": color,
                "size": self.np_random.integers(2, 5)
            })

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
        print("✓ Running implementation validation...")
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

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
        
        # Transpose the observation back for Pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Wait 3 seconds before resetting
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()