
# Generated: 2025-08-27T13:22:52.053082
# Source Brief: brief_00348.md
# Brief Index: 348

        
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
        "Controls: Use arrow keys to select a tower position. Press space to build a tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist tower defense game. Survive three waves of enemies by strategically placing towers."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 3000 # Increased for a longer game

    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_PATH = (50, 50, 70)
    COLOR_BASE = (60, 180, 75)
    COLOR_BASE_STROKE = (100, 220, 115)
    COLOR_ENEMY = (210, 60, 60)
    COLOR_TOWER = (60, 150, 210)
    COLOR_PROJECTILE = (255, 225, 25)
    COLOR_SELECTOR_VALID = (255, 255, 255, 100)
    COLOR_SELECTOR_INVALID = (255, 0, 0, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_MONEY = (255, 215, 0)
    COLOR_HEALTH = (230, 60, 60)

    # Game Parameters
    BASE_HEALTH_START = 100
    STARTING_MONEY = 150
    ENEMY_KILL_REWARD = 25
    
    TOWER_SPECS = [
        {"cost": 100, "range": 80, "fire_rate": 0.8, "color": COLOR_TOWER, "name": "Basic Turret"},
    ]

    WAVE_DATA = [
        {"count": 10, "speed": 1.0, "health": 100, "spawn_delay": 45},
        {"count": 15, "speed": 1.2, "health": 120, "spawn_delay": 30},
        {"count": 25, "speed": 1.4, "health": 150, "spawn_delay": 20},
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
        self.font_small = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Define game geometry
        self._define_geometry()
        
        # Initialize state variables
        self.reset()
        
        # Self-check
        self.validate_implementation()
    
    def _define_geometry(self):
        """Defines the static geometry of the game like path and tower spots."""
        self.path_waypoints = [
            pygame.math.Vector2(-50, 200),
            pygame.math.Vector2(120, 200),
            pygame.math.Vector2(120, 80),
            pygame.math.Vector2(320, 80),
            pygame.math.Vector2(320, 320),
            pygame.math.Vector2(520, 320),
            pygame.math.Vector2(520, 150),
        ]
        self.base_pos = pygame.math.Vector2(520, 150)
        self.base_rect = pygame.Rect(self.base_pos.x - 20, self.base_pos.y - 20, 40, 40)
        
        self.tower_spots = [
            pygame.math.Vector2(120, 140),
            pygame.math.Vector2(180, 80),
            pygame.math.Vector2(260, 140),
            pygame.math.Vector2(320, 260),
            pygame.math.Vector2(450, 260),
            pygame.math.Vector2(520, 230),
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = self.BASE_HEALTH_START
        self.money = self.STARTING_MONEY
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.current_wave_index = -1
        self.wave_timer = 120  # Time before first wave
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        
        self.selector_index = 0
        self.selected_tower_type = 0 # Only one type for now
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.current_wave_index += 1
        if self.current_wave_index >= len(self.WAVE_DATA):
            self.game_won = True
            return
            
        wave_info = self.WAVE_DATA[self.current_wave_index]
        self.enemies_to_spawn = wave_info["count"]
        self.spawn_timer = 0
    
    def step(self, action):
        reward_buffer = 0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        # --- Handle player input ---
        # Cycle selector
        if movement in [1, 3]: # Up or Left
            self.selector_index = (self.selector_index - 1 + len(self.tower_spots)) % len(self.tower_spots)
        elif movement in [2, 4]: # Down or Right
            self.selector_index = (self.selector_index + 1) % len(self.tower_spots)
            
        # Place tower (on key press, not hold)
        if space_held and not self.prev_space_held:
            reward_buffer += self._place_tower()

        # Cycle tower type (on key press, not hold)
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # --- Update game logic ---
        if not self.game_over and not self.game_won:
            self.steps += 1
            
            # Wave Management
            if self.enemies_to_spawn > 0:
                self.spawn_timer -= 1
                if self.spawn_timer <= 0:
                    self._spawn_enemy()
                    self.spawn_timer = self.WAVE_DATA[self.current_wave_index]["spawn_delay"]
            elif len(self.enemies) == 0:
                # Wave cleared
                if self.current_wave_index < len(self.WAVE_DATA):
                    reward_buffer += 10 # Wave clear reward
                    # SFX: Wave Cleared
                    self._start_next_wave()

            # Update Entities
            reward_buffer += self._update_enemies()
            self._update_towers()
            reward_buffer += self._update_projectiles()
            self._update_particles()
        
        # --- Check termination conditions ---
        if self.base_health <= 0 and not self.game_over:
            self.game_over = True
            # SFX: Game Over
        
        if self.game_won and len(self.enemies) == 0 and len(self.projectiles) == 0:
            reward_buffer += 50 # Game win reward
            self.game_over = True # End the game on win
            
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        self.score += reward_buffer
        
        return (
            self._get_observation(),
            reward_buffer,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _place_tower(self):
        spot_pos = self.tower_spots[self.selector_index]
        spec = self.TOWER_SPECS[self.selected_tower_type]
        
        # Check if spot is occupied
        is_occupied = any(tower['pos'] == spot_pos for tower in self.towers)
        if is_occupied:
            # SFX: Action Failed
            return 0
        
        # Check if enough money
        if self.money >= spec['cost']:
            self.money -= spec['cost']
            self.towers.append({
                "pos": spot_pos,
                "spec": spec,
                "cooldown": 0,
                "target": None
            })
            # SFX: Tower Placed
            self._create_particles(spot_pos, 20, spec['color'], 2, 4, 15)
            return 0.5 # Small reward for building
        else:
            # SFX: Not enough money
            return 0

    def _spawn_enemy(self):
        wave_info = self.WAVE_DATA[self.current_wave_index]
        self.enemies.append({
            "pos": self.path_waypoints[0].copy(),
            "health": wave_info["health"],
            "max_health": wave_info["health"],
            "speed": wave_info["speed"],
            "waypoint_index": 1
        })
        self.enemies_to_spawn -= 1

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy["waypoint_index"] >= len(self.path_waypoints):
                self.base_health -= 10
                self.enemies.remove(enemy)
                reward -= 1 # Penalty for reaching base
                # SFX: Base Damaged
                self._create_particles(self.base_pos, 30, self.COLOR_HEALTH, 3, 6, 20)
                continue
                
            target_pos = self.path_waypoints[enemy["waypoint_index"]]
            direction = (target_pos - enemy["pos"])
            
            if direction.length() < enemy["speed"]:
                enemy["pos"] = target_pos
                enemy["waypoint_index"] += 1
            else:
                enemy["pos"] += direction.normalize() * enemy["speed"]
        return reward

    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue
            
            # Find a target
            possible_targets = []
            for enemy in self.enemies:
                dist = tower["pos"].distance_to(enemy["pos"])
                if dist <= tower["spec"]["range"]:
                    possible_targets.append(enemy)
            
            if possible_targets:
                # Target the enemy closest to the end of the path
                target = max(possible_targets, key=lambda e: e["waypoint_index"] + e["pos"].distance_to(self.path_waypoints[e["waypoint_index"]]))
                
                self.projectiles.append({
                    "start_pos": tower["pos"].copy(),
                    "pos": tower["pos"].copy(),
                    "target_enemy": target,
                    "speed": 8
                })
                tower["cooldown"] = self.FPS / tower["spec"]["fire_rate"]
                # SFX: Tower Fire

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            target = proj["target_enemy"]
            
            if target not in self.enemies: # Target already dead
                self.projectiles.remove(proj)
                continue
            
            direction = (target["pos"] - proj["pos"]).normalize()
            proj["pos"] += direction * proj["speed"]
            
            if proj["pos"].distance_to(target["pos"]) < 5:
                # Hit
                damage = 50
                target["health"] -= damage
                reward += 0.1 # Reward for hit
                self._create_particles(proj["pos"], 5, self.COLOR_PROJECTILE, 1, 2, 8)
                self.projectiles.remove(proj)
                # SFX: Enemy Hit
                
                if target["health"] <= 0:
                    reward += 1 # Reward for kill
                    self.money += self.ENEMY_KILL_REWARD
                    self._create_particles(target["pos"], 25, self.COLOR_ENEMY, 2, 5, 25)
                    self.enemies.remove(target)
                    # SFX: Enemy Destroyed
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, min_speed, max_speed, max_life):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_speed, max_speed)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed / self.FPS
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": random.randint(max_life // 2, max_life),
                "max_life": max_life,
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_path()
        self._render_placement_spots()
        self._render_selector()
        self._render_base()
        self._render_enemies()
        self._render_towers()
        self._render_projectiles()
        self._render_particles()

        if self.game_over and not self.game_won:
            self._render_text_centered("GAME OVER", self.font_large, (255, 80, 80))
        elif self.game_won:
             self._render_text_centered("YOU WIN!", self.font_large, (80, 255, 80))

    def _render_text_centered(self, text, font, color):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def _render_path(self):
        if len(self.path_waypoints) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, [(int(p.x), int(p.y)) for p in self.path_waypoints], 30)

    def _render_base(self):
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.base_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_BASE_STROKE, self.base_rect, 2, border_radius=4)

    def _render_placement_spots(self):
        for i, spot in enumerate(self.tower_spots):
            is_occupied = any(tower['pos'] == spot for tower in self.towers)
            color = (100, 100, 120, 50) if is_occupied else (255, 255, 255, 20)
            pygame.gfxdraw.filled_circle(self.screen, int(spot.x), int(spot.y), 15, color)
            pygame.gfxdraw.aacircle(self.screen, int(spot.x), int(spot.y), 15, color)
    
    def _render_selector(self):
        if self.game_over: return
        spot_pos = self.tower_spots[self.selector_index]
        is_occupied = any(tower['pos'] == spot_pos for tower in self.towers)
        can_afford = self.money >= self.TOWER_SPECS[self.selected_tower_type]['cost']
        
        color = self.COLOR_SELECTOR_VALID
        if is_occupied or not can_afford:
            color = self.COLOR_SELECTOR_INVALID

        radius = 20 + 3 * math.sin(self.steps * 0.2)
        pygame.gfxdraw.aacircle(self.screen, int(spot_pos.x), int(spot_pos.y), int(radius), color)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos_x, pos_y = int(enemy["pos"].x), int(enemy["pos"].y)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, 8, tuple(min(255, c+50) for c in self.COLOR_ENEMY))
            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            bar_width = int(14 * health_pct)
            pygame.draw.rect(self.screen, (0,0,0), (pos_x - 8, pos_y - 14, 16, 4))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH, (pos_x - 7, pos_y - 13, bar_width, 2))

    def _render_towers(self):
        for tower in self.towers:
            pos = tower["pos"]
            color = tower["spec"]["color"]
            
            # Simple triangle shape for tower
            points = [
                (pos.x, pos.y - 10),
                (pos.x - 8, pos.y + 6),
                (pos.x + 8, pos.y + 6)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            
            # Range indicator
            if tower["pos"] == self.tower_spots[self.selector_index]:
                 pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), tower['spec']['range'], (255,255,255, 30))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(proj['pos'].x), int(proj['pos'].y)), 3)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(3 * (p['life'] / p['max_life']))
            if size > 0:
                # Create a temporary surface for alpha blending
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size, size), size)
                self.screen.blit(particle_surf, (int(p['pos'].x - size), int(p['pos'].y - size)))

    def _render_ui(self):
        # Health
        health_text = self.font_small.render(f"Base Health: {int(self.base_health)}%", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Money
        money_text = self.font_small.render(f"Money: ${self.money}", True, self.COLOR_MONEY)
        self.screen.blit(money_text, (10, 30))

        # Wave
        wave_str = f"Wave: {self.current_wave_index + 1}/{len(self.WAVE_DATA)}"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))
        
        # Enemies left
        enemies_left = len(self.enemies) + self.enemies_to_spawn
        enemies_text = self.font_small.render(f"Enemies: {enemies_left}", True, self.COLOR_TEXT)
        self.screen.blit(enemies_text, (self.SCREEN_WIDTH - enemies_text.get_width() - 10, 30))

        # Score
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH / 2 - score_text.get_width() / 2, 10))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "money": self.money,
            "current_wave": self.current_wave_index + 1,
            "towers_built": len(self.towers),
        }
        
    def close(self):
        pygame.font.quit()
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    # Set SDL_VIDEODRIVER to "dummy" for headless execution
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Tower Defense")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        # --- Human Input ---
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(GameEnv.FPS)

    env.close()