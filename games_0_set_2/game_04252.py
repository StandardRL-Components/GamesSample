
# Generated: 2025-08-28T01:51:20.161362
# Source Brief: brief_04252.md
# Brief Index: 4252

        
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
        "Controls: Use arrows to move the placement cursor. "
        "Press Shift to cycle tower types. Press Space to build a tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of invaders by placing defensive towers "
        "on the battlefield. Manage your resources and survive all waves to win."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True

    # --- Colors ---
    COLOR_BG = (18, 23, 37)
    COLOR_PATH = (40, 50, 70)
    COLOR_PATH_BORDER = (60, 75, 105)
    COLOR_GRID = (70, 85, 115, 100)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_BASE = (0, 150, 200)
    COLOR_BASE_BORDER = (0, 190, 255)
    
    COLOR_ENEMY = (217, 3, 104)
    COLOR_ENEMY_BORDER = (255, 80, 150)
    COLOR_ENEMY_STRONG = (255, 107, 0)
    COLOR_ENEMY_STRONG_BORDER = (255, 165, 0)
    
    COLOR_TEXT = (230, 230, 230)
    COLOR_HEALTH_BAR_BG = (70, 20, 20)
    COLOR_HEALTH_BAR_FG = (200, 30, 30)
    COLOR_BASE_HEALTH_FG = (0, 200, 100)
    
    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 30 * 120 # 2 minutes at 30fps
    
    BASE_START_HEALTH = 100
    STARTING_RESOURCES = 250
    NUM_WAVES = 5

    ISO_TILE_WIDTH = 64
    ISO_TILE_HEIGHT = 32
    ISO_ORIGIN_X = SCREEN_WIDTH // 2
    ISO_ORIGIN_Y = 80

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
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_huge = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Game data setup
        self._setup_game_data()

        # Initialize state variables by calling reset
        self.reset()

        # Run validation check
        # self.validate_implementation() # Commented out for final submission

    def _setup_game_data(self):
        # --- Path Definition (in grid coordinates) ---
        self.path_waypoints = [
            (-5, 4), (-4, 4), (-3, 4), (-2, 4), (-1, 4), (0, 4), (1, 4),
            (1, 3), (1, 2), (1, 1), (1, 0), (1, -1), (1, -2), (1, -3),
            (2, -3), (3, -3), (4, -3), (5, -3), (6, -3)
        ]

        # --- Tower Placement Grid ---
        self.grid_locs = [
            (x, y) for x in range(-2, 3) for y in range(-2, 3)
            if (x, y) not in [(-1, 1), (0, 0), (1, -1)] # Exclude path-like areas
        ]
        self.grid_width = 5

        # --- Tower Types ---
        self.tower_types = [
            {"name": "Cannon", "cost": 100, "range": 3.0, "damage": 15, "fire_rate": 30, "color": (100, 255, 100), "proj_speed": 8},
            {"name": "Missile", "cost": 175, "range": 4.5, "damage": 40, "fire_rate": 90, "color": (255, 165, 0), "proj_speed": 5},
        ]
        
        # --- Wave Data: (num_enemies, base_health, base_speed, spawn_delay) ---
        self.wave_data = [
            (10, 50, 1.0, 45),
            (15, 60, 1.1, 40),
            (20, 75, 1.2, 35),
            (10, 150, 1.3, 60), # Stronger enemies
            (30, 90, 1.4, 25),
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.reward_buffer = 0
        self.game_over = False
        self.game_won = False

        self.base_health = self.BASE_START_HEALTH
        self.resources = self.STARTING_RESOURCES
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.current_wave = -1
        self.wave_timer = 90 # Time until first wave
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        
        self.cursor_grid_pos = [2, 2] # Center of 5x5 grid
        self.selected_tower_type = 0
        
        self.last_shift_press = False
        self.last_space_press = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.reward_buffer = 0
        self.steps += 1
        
        if self.auto_advance:
            self.clock.tick(30)

        if not self.game_over:
            self._handle_input(action)
            self._update_waves()
            self._update_towers()
            self._update_projectiles()
            self._update_enemies()
        
        self._update_particles()
        
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            self.reward_buffer,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 1: self.cursor_grid_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_grid_pos[1] += 1 # Down
        elif movement == 3: self.cursor_grid_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_grid_pos[0] += 1 # Right
        self.cursor_grid_pos[0] = np.clip(self.cursor_grid_pos[0], 0, self.grid_width - 1)
        self.cursor_grid_pos[1] = np.clip(self.cursor_grid_pos[1], 0, self.grid_width - 1)

        # --- Cycle Tower (Edge-triggered) ---
        if shift_held and not self.last_shift_press:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_types)
            # sfx: menu_cycle.wav
        self.last_shift_press = shift_held

        # --- Place Tower (Edge-triggered) ---
        if space_held and not self.last_space_press:
            self._place_tower()
        self.last_space_press = space_held

    def _place_tower(self):
        grid_idx = self.cursor_grid_pos[1] * self.grid_width + self.cursor_grid_pos[0]
        if grid_idx >= len(self.grid_locs): return

        grid_pos = self.grid_locs[grid_idx]
        tower_spec = self.tower_types[self.selected_tower_type]

        # Check cost and if location is occupied
        if self.resources >= tower_spec["cost"]:
            is_occupied = any(t['grid_pos'] == grid_pos for t in self.towers)
            if not is_occupied:
                self.resources -= tower_spec["cost"]
                self.towers.append({
                    "grid_pos": grid_pos,
                    "type_idx": self.selected_tower_type,
                    "cooldown": 0,
                })
                # sfx: build_tower.wav
                self._create_particles(self._iso_to_screen(*grid_pos), 10, (200, 200, 255), 15)


    def _update_waves(self):
        if self.current_wave == self.NUM_WAVES: return

        # If wave is over and all enemies are gone, start timer for next wave
        if self.enemies_to_spawn == 0 and not self.enemies:
            if self.current_wave > -1:
                self.reward_buffer += 10
                # sfx: wave_complete.wav
            
            self.current_wave += 1
            if self.current_wave == self.NUM_WAVES:
                self.game_won = True
                return
            
            self.wave_timer = 150 # 5 seconds
            wave_info = self.wave_data[self.current_wave]
            self.enemies_to_spawn = wave_info[0]

        # Countdown to start spawning
        if self.wave_timer > 0:
            self.wave_timer -= 1
            return

        # Spawn enemies
        if self.enemies_to_spawn > 0 and self.spawn_timer <= 0:
            wave_info = self.wave_data[self.current_wave]
            spawn_delay = wave_info[3]
            
            self.spawn_timer = spawn_delay
            self.enemies_to_spawn -= 1
            
            difficulty_mod = 1.0 + self.current_wave * 0.1
            start_pos = self.path_waypoints[0]
            
            self.enemies.append({
                "pos": list(self._iso_to_screen(start_pos[0] - 0.5, start_pos[1] - 0.5)),
                "max_health": wave_info[1] * difficulty_mod,
                "health": wave_info[1] * difficulty_mod,
                "speed": wave_info[2] * difficulty_mod,
                "waypoint_idx": 1,
                "hit_timer": 0,
            })
            # sfx: enemy_spawn.wav

        if self.spawn_timer > 0:
            self.spawn_timer -= 1

    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            if enemy["hit_timer"] > 0: enemy["hit_timer"] -= 1

            if enemy["waypoint_idx"] >= len(self.path_waypoints):
                self.base_health -= 10
                self.reward_buffer -= 0.1
                self.enemies.remove(enemy)
                self._create_particles(self._iso_to_screen(6.5, -2.5), 15, self.COLOR_ENEMY, 20)
                # sfx: base_damage.wav
                continue
            
            target_grid = self.path_waypoints[enemy["waypoint_idx"]]
            target_pos = self._iso_to_screen(target_grid[0], target_grid[1])
            
            dx = target_pos[0] - enemy["pos"][0]
            dy = target_pos[1] - enemy["pos"][1]
            dist = math.hypot(dx, dy)
            
            if dist < enemy["speed"]:
                enemy["waypoint_idx"] += 1
            else:
                enemy["pos"][0] += (dx / dist) * enemy["speed"]
                enemy["pos"][1] += (dy / dist) * enemy["speed"]

    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue

            spec = self.tower_types[tower["type_idx"]]
            tower_pos = self._iso_to_screen(*tower["grid_pos"])
            
            # Find target
            target = None
            min_dist = spec["range"] * self.ISO_TILE_WIDTH * 0.5 # Range in pixels
            
            for enemy in self.enemies:
                dist = math.hypot(enemy["pos"][0] - tower_pos[0], enemy["pos"][1] - tower_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                tower["cooldown"] = spec["fire_rate"]
                self.projectiles.append({
                    "pos": list(tower_pos),
                    "target": target,
                    "spec": spec,
                })
                # sfx: cannon_fire.wav or missile_launch.wav
                self._create_particles(tower_pos, 3, (255, 255, 200), 5, 2)


    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            spec = proj["spec"]
            target = proj["target"]

            dx = target["pos"][0] - proj["pos"][0]
            dy = target["pos"][1] - proj["pos"][1]
            dist = math.hypot(dx, dy)

            if dist < spec["proj_speed"]:
                # Hit
                target["health"] -= spec["damage"]
                target["hit_timer"] = 5 # Flash for 5 frames
                self.reward_buffer += 0.1
                self.projectiles.remove(proj)
                self._create_particles(target["pos"], 5, spec["color"], 10)
                # sfx: enemy_hit.wav

                if target["health"] <= 0:
                    self.score += 10
                    self.reward_buffer += 1
                    self.resources += 15
                    if target in self.enemies: self.enemies.remove(target)
                    self._create_particles(target["pos"], 20, self.COLOR_ENEMY_BORDER, 25)
                    # sfx: enemy_explode.wav
            else:
                # Move towards target
                proj["pos"][0] += (dx / dist) * spec["proj_speed"]
                proj["pos"][1] += (dy / dist) * spec["proj_speed"]

    def _update_particles(self):
        for p in reversed(self.particles):
            p['life'] -= 1
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.game_over:
            return True

        if self.base_health <= 0:
            self.game_over = True
            self.reward_buffer -= 100
            self.score -= 1000
            # sfx: game_over.wav
            return True
        
        if self.game_won:
            self.game_over = True
            self.reward_buffer += 100
            self.score += 5000
            # sfx: victory.wav
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

    def _render_game(self):
        # Render path
        for i in range(len(self.path_waypoints) - 1):
            p1 = self._iso_to_screen(*self.path_waypoints[i])
            p2 = self._iso_to_screen(*self.path_waypoints[i+1])
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, self.ISO_TILE_HEIGHT)
        for p in self.path_waypoints:
            pos = self._iso_to_screen(*p)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ISO_TILE_HEIGHT // 2, self.COLOR_PATH)
        
        # Render base
        base_pos = self._iso_to_screen(7, -3)
        pygame.gfxdraw.filled_circle(self.screen, base_pos[0], base_pos[1], 20, self.COLOR_BASE_BORDER)
        pygame.gfxdraw.filled_circle(self.screen, base_pos[0], base_pos[1], 18, self.COLOR_BASE)

        # Render grid and cursor
        self._render_grid()

        # Render towers
        for tower in self.towers:
            spec = self.tower_types[tower["type_idx"]]
            pos = self._iso_to_screen(*tower["grid_pos"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, (50,50,50))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, spec["color"])
            
        # Render enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            color = self.COLOR_ENEMY if enemy['max_health'] < 100 else self.COLOR_ENEMY_STRONG
            border_color = self.COLOR_ENEMY_BORDER if enemy['max_health'] < 100 else self.COLOR_ENEMY_STRONG_BORDER
            
            if enemy["hit_timer"] > 0:
                color, border_color = (255, 255, 255), (255, 255, 255)
            
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, border_color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, color)
            
            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            bar_w = 16
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (pos[0] - bar_w/2, pos[1] - 15, bar_w, 4))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (pos[0] - bar_w/2, pos[1] - 15, bar_w * health_pct, 4))

        # Render projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            color = proj["spec"]["color"]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, color)

        # Render particles
        for p in self.particles:
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, p['color'])

    def _render_grid(self):
        cursor_idx = self.cursor_grid_pos[1] * self.grid_width + self.cursor_grid_pos[0]
        
        for i, grid_pos in enumerate(self.grid_locs):
            is_cursor = (i == cursor_idx)
            is_occupied = any(t['grid_pos'] == grid_pos for t in self.towers)
            
            color = self.COLOR_GRID
            if is_cursor:
                color = self.COLOR_CURSOR
            elif is_occupied:
                color = (100, 30, 30, 150)
            
            self._draw_iso_tile(grid_pos, color, filled=is_cursor)

    def _render_ui(self):
        # Top bar background
        ui_bar = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_bar.fill((10, 15, 25, 200))
        self.screen.blit(ui_bar, (0, 0))

        # Base Health
        health_text = self.font_large.render(f"Base: {max(0, self.base_health)}%", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 8))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (120, 12, 100, 16))
        health_pct = max(0, self.base_health / self.BASE_START_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_BASE_HEALTH_FG, (120, 12, 100 * health_pct, 16))

        # Resources
        res_text = self.font_large.render(f"Gold: {self.resources}", True, (255, 215, 0))
        self.screen.blit(res_text, (240, 8))

        # Wave
        wave_str = f"Wave: {self.current_wave + 1}/{self.NUM_WAVES}"
        if self.current_wave == -1: wave_str = "Starting..."
        elif self.game_won: wave_str = "VICTORY!"
        wave_text = self.font_large.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - 180, 8))
        
        # Selected Tower Info
        spec = self.tower_types[self.selected_tower_type]
        can_afford = self.resources >= spec['cost']
        color = self.COLOR_TEXT if can_afford else self.COLOR_HEALTH_BAR_FG
        tower_info_text = self.font_small.render(
            f"Build: {spec['name']} (Cost: {spec['cost']})", True, color
        )
        self.screen.blit(tower_info_text, (10, self.SCREEN_HEIGHT - 26))

        # Game Over / Win Text
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "VICTORY!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 50, 50)
            end_text = self.font_huge.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave + 1,
        }

    # --- Helper Functions ---
    def _iso_to_screen(self, x, y):
        screen_x = self.ISO_ORIGIN_X + (x - y) * self.ISO_TILE_WIDTH / 2
        screen_y = self.ISO_ORIGIN_Y + (x + y) * self.ISO_TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, grid_pos, color, filled=False):
        x, y = grid_pos
        points = [
            self._iso_to_screen(x, y),
            self._iso_to_screen(x + 1, y),
            self._iso_to_screen(x + 1, y + 1),
            self._iso_to_screen(x, y + 1)
        ]
        if filled:
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        else:
            pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _create_particles(self, pos, count, color, life, speed=4):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            s = random.uniform(0.5, 1) * speed
            vel = [math.cos(angle) * s, math.sin(angle) * s]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': random.randint(life // 2, life),
                'max_life': life,
                'size': random.randint(2, 5),
                'color': color
            })

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    while not terminated:
        # --- Human Input to Action ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # --- Render to Display ---
        # The observation is (H, W, C), but pygame blit needs a surface
        # So we'll just use the env's internal screen
        surf = pygame.transform.rotate(env.screen, -90)
        surf = pygame.transform.flip(surf, True, False)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    pygame.quit()