
# Generated: 2025-08-28T01:50:56.981979
# Source Brief: brief_04250.md
# Brief Index: 4250

        
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
        "Controls: Arrow keys to move your robot. Press space to fire your weapon. "
        "Evade enemy fire and destroy all hostiles."
    )

    game_description = (
        "Pilot a combat robot in a grid-based arena. Blast through waves of enemies "
        "in this fast-paced, top-down arcade shooter. Your goal is ultimate robotic domination."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500 # 50 seconds at 30fps

        # Grid and cell dimensions
        self.GRID_COLS, self.GRID_ROWS = 12, 7
        self.CELL_SIZE = 50
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_ACCENT = (200, 255, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_ACCENT = (255, 180, 180)
        self.COLOR_PLAYER_PROJ = (255, 255, 0)
        self.COLOR_ENEMY_PROJ = (255, 150, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_EXPLOSION = [(255, 200, 0), (255, 150, 0), (255, 100, 0)]

        # Game parameters
        self.PLAYER_MAX_HEALTH = 5
        self.PLAYER_SPEED = 6.0  # pixels per frame
        self.PLAYER_SHOOT_COOLDOWN = 6  # frames
        self.PLAYER_INVULN_DURATION = 60 # frames
        self.PROJECTILE_SPEED = 12.0
        self.TOTAL_ENEMIES = 10
        self.BASE_ENEMY_SPEED = 1.0
        self.ENEMY_SPEED_INCREMENT = 0.2
        self.ENEMY_SHOOT_COOLDOWN = 45 # frames
        self.ENEMY_CHASE_DIST = 5 # grid cells

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.player = {}
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.space_was_held = False
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            # Fallback for older gym versions or no seed
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.enemies_defeated_count = 0
        
        # Player state
        start_pos = [self.GRID_COLS // 2, self.GRID_ROWS - 1]
        self.player = {
            "grid_pos": start_pos,
            "pixel_pos": self._grid_to_pixel(start_pos),
            "health": self.PLAYER_MAX_HEALTH,
            "shoot_cooldown": 0,
            "invuln_timer": self.PLAYER_INVULN_DURATION,
            "last_move_dir": (0, -1), # Start aiming up
            "radius": self.CELL_SIZE * 0.35,
        }

        # Enemy state
        self.enemies = []
        spawn_points = self._get_spawn_points(self.TOTAL_ENEMIES, [start_pos])
        for pos in spawn_points:
            self.enemies.append({
                "grid_pos": pos,
                "pixel_pos": self._grid_to_pixel(pos),
                "target_pixel_pos": self._grid_to_pixel(pos),
                "ai_state": "PATROL",
                "ai_timer": self.np_random.integers(30, 90),
                "shoot_cooldown": self.np_random.integers(30, self.ENEMY_SHOOT_COOLDOWN),
                "radius": self.CELL_SIZE * 0.3,
            })
        
        self.projectiles.clear()
        self.particles.clear()
        self.space_was_held = True # Prevent shooting on first frame
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.reward_this_step = -0.01 # Small time penalty

        self._handle_input(action)
        self._update_player()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        
        terminated = self._check_termination()
        
        self.steps += 1
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        move_dir = [0, 0]
        if movement == 1: move_dir[1] = -1  # Up
        elif movement == 2: move_dir[1] = 1   # Down
        elif movement == 3: move_dir[0] = -1  # Left
        elif movement == 4: move_dir[0] = 1   # Right
        
        if movement != 0:
            self.player["last_move_dir"] = tuple(move_dir)
            
        move_vec = np.array(move_dir, dtype=np.float32)
        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec) * self.PLAYER_SPEED
        
        self.player["pixel_pos"][0] += move_vec[0]
        self.player["pixel_pos"][1] += move_vec[1]
        
        # Clamp player to grid area
        self.player["pixel_pos"][0] = np.clip(self.player["pixel_pos"][0], self.GRID_X_OFFSET, self.GRID_X_OFFSET + self.GRID_WIDTH)
        self.player["pixel_pos"][1] = np.clip(self.player["pixel_pos"][1], self.GRID_Y_OFFSET, self.GRID_Y_OFFSET + self.GRID_HEIGHT)
        self.player["grid_pos"] = self._pixel_to_grid(self.player["pixel_pos"])

        # Shooting
        if space_held and not self.space_was_held and self.player["shoot_cooldown"] <= 0:
            # sfx: player_shoot.wav
            self.player["shoot_cooldown"] = self.PLAYER_SHOOT_COOLDOWN
            proj_vel = np.array(self.player["last_move_dir"], dtype=np.float32) * self.PROJECTILE_SPEED
            self.projectiles.append({
                "pos": list(self.player["pixel_pos"]),
                "vel": list(proj_vel),
                "owner": "player",
                "radius": 5,
                "color": self.COLOR_PLAYER_PROJ
            })
        self.space_was_held = space_held

    def _update_player(self):
        if self.player["shoot_cooldown"] > 0:
            self.player["shoot_cooldown"] -= 1
        if self.player["invuln_timer"] > 0:
            self.player["invuln_timer"] -= 1

    def _update_enemies(self):
        current_enemy_speed = self.BASE_ENEMY_SPEED + (self.enemies_defeated_count // 2) * self.ENEMY_SPEED_INCREMENT

        for enemy in self.enemies:
            # AI Logic
            enemy["ai_timer"] -= 1
            dist_to_player = self._grid_dist(enemy["grid_pos"], self.player["grid_pos"])
            
            if dist_to_player <= self.ENEMY_CHASE_DIST:
                enemy["ai_state"] = "CHASE"
            elif enemy["ai_state"] == "CHASE": # Lost player
                enemy["ai_state"] = "PATROL"
                enemy["ai_timer"] = 0

            if enemy["ai_timer"] <= 0:
                if enemy["ai_state"] == "PATROL":
                    enemy["ai_timer"] = self.np_random.integers(45, 120)
                    possible_moves = []
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        new_pos = [enemy["grid_pos"][0] + dx, enemy["grid_pos"][1] + dy]
                        if 0 <= new_pos[0] < self.GRID_COLS and 0 <= new_pos[1] < self.GRID_ROWS:
                            possible_moves.append(new_pos)
                    if possible_moves:
                        target_grid_pos = random.choice(possible_moves)
                        enemy["target_pixel_pos"] = self._grid_to_pixel(target_grid_pos)
                
                elif enemy["ai_state"] == "CHASE":
                    enemy["target_pixel_pos"] = list(self.player["pixel_pos"])


            # Movement
            direction = np.array(enemy["target_pixel_pos"]) - np.array(enemy["pixel_pos"])
            norm = np.linalg.norm(direction)
            if norm > 1:
                direction = direction / norm
                enemy["pixel_pos"][0] += direction[0] * current_enemy_speed
                enemy["pixel_pos"][1] += direction[1] * current_enemy_speed
            enemy["grid_pos"] = self._pixel_to_grid(enemy["pixel_pos"])

            # Shooting
            enemy["shoot_cooldown"] -= 1
            if enemy["shoot_cooldown"] <= 0 and enemy["ai_state"] == "CHASE":
                # sfx: enemy_shoot.wav
                enemy["shoot_cooldown"] = self.ENEMY_SHOOT_COOLDOWN + self.np_random.integers(-15, 15)
                proj_vel = np.array(self.player["pixel_pos"]) - np.array(enemy["pixel_pos"])
                norm = np.linalg.norm(proj_vel)
                if norm > 0:
                    proj_vel = proj_vel / norm * self.PROJECTILE_SPEED * 0.75
                    self.projectiles.append({
                        "pos": list(enemy["pixel_pos"]), "vel": list(proj_vel),
                        "owner": "enemy", "radius": 4, "color": self.COLOR_ENEMY_PROJ
                    })

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj["pos"][0] += proj["vel"][0]
            proj["pos"][1] += proj["vel"][1]
            
            # Off-screen check
            if not (0 < proj["pos"][0] < self.WIDTH and 0 < proj["pos"][1] < self.HEIGHT):
                self.projectiles.remove(proj)
                continue
            
            # Collision check
            if proj["owner"] == "player":
                for enemy in self.enemies[:]:
                    if self._check_collision(proj, enemy):
                        # sfx: hit_confirm.wav
                        self._spawn_explosion(enemy["pixel_pos"], 15, self.COLOR_EXPLOSION)
                        self.enemies.remove(enemy)
                        self.projectiles.remove(proj)
                        self.score += 10
                        self.reward_this_step += 1.0 # Defeat reward
                        self.enemies_defeated_count += 1
                        break
            elif proj["owner"] == "enemy":
                if self.player["invuln_timer"] <= 0 and self._check_collision(proj, self.player):
                    # sfx: player_hit.wav
                    self.player["health"] -= 1
                    self.player["invuln_timer"] = self.PLAYER_INVULN_DURATION
                    self.reward_this_step -= 0.5 # Penalty for getting hit
                    self._spawn_explosion(self.player["pixel_pos"], 10, [self.COLOR_PLAYER, self.COLOR_PLAYER_ACCENT])
                    self.projectiles.remove(proj)
                    break

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.player["health"] <= 0:
            self.game_over = True
            self.reward_this_step -= 100 # Loss penalty
            # sfx: game_over.wav
        elif not self.enemies:
            self.game_over = True
            self.score += 1000 # Victory bonus
            self.reward_this_step += 100 # Win reward
            # sfx: victory.wav
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.reward_this_step -= 50 # Timeout penalty
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_grid(self):
        for i in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT))
        for i in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH, y))

    def _render_game_elements(self):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / p["max_lifetime"]))
            p_color = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["size"]), p_color)

        # Projectiles
        for proj in self.projectiles:
            x, y = int(proj["pos"][0]), int(proj["pos"][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, proj["radius"], proj["color"])
            pygame.gfxdraw.aacircle(self.screen, x, y, proj["radius"], proj["color"])

        # Enemies
        for enemy in self.enemies:
            self._draw_robot_sprite(self.screen, enemy["pixel_pos"], enemy["radius"], self.COLOR_ENEMY, self.COLOR_ENEMY_ACCENT, (0,-1))
            
        # Player
        is_visible = self.player["invuln_timer"] <= 0 or (self.player["invuln_timer"] // 3) % 2 == 0
        if is_visible:
            self._draw_robot_sprite(self.screen, self.player["pixel_pos"], self.player["radius"], self.COLOR_PLAYER, self.COLOR_PLAYER_ACCENT, self.player["last_move_dir"])

    def _draw_robot_sprite(self, surface, pos, radius, color, accent_color, direction):
        x, y = int(pos[0]), int(pos[1])
        # Body
        pygame.gfxdraw.filled_circle(surface, x, y, int(radius), color)
        pygame.gfxdraw.aacircle(surface, x, y, int(radius), color)
        # Cockpit
        pygame.gfxdraw.filled_circle(surface, x, y, int(radius * 0.6), accent_color)
        # Turret
        dir_vec = np.array(direction)
        start_point = (x, y)
        end_point = (x + dir_vec[0] * radius * 1.2, y + dir_vec[1] * radius * 1.2)
        pygame.draw.line(surface, accent_color, start_point, end_point, 4)

    def _render_ui(self):
        # Health
        health_text = self.font_ui.render(f"HEALTH: {self.player['health']}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Enemies remaining
        enemy_text = self.font_ui.render(f"ENEMIES: {len(self.enemies)}", True, self.COLOR_TEXT)
        self.screen.blit(enemy_text, (self.WIDTH // 2 - enemy_text.get_width() // 2, 10))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY" if not self.enemies else "GAME OVER"
            color = (100, 255, 100) if not self.enemies else (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player["health"],
            "enemies_remaining": len(self.enemies),
        }

    # --- Helper Methods ---

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_X_OFFSET + (grid_pos[0] + 0.5) * self.CELL_SIZE
        y = self.GRID_Y_OFFSET + (grid_pos[1] + 0.5) * self.CELL_SIZE
        return [x, y]

    def _pixel_to_grid(self, pixel_pos):
        gx = int((pixel_pos[0] - self.GRID_X_OFFSET) / self.CELL_SIZE)
        gy = int((pixel_pos[1] - self.GRID_Y_OFFSET) / self.CELL_SIZE)
        return [max(0, min(self.GRID_COLS - 1, gx)), max(0, min(self.GRID_ROWS - 1, gy))]

    def _grid_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _check_collision(self, obj1, obj2):
        dist_sq = (obj1["pos"][0] - obj2["pixel_pos"][0])**2 + (obj1["pos"][1] - obj2["pixel_pos"][1])**2
        return dist_sq < (obj1["radius"] + obj2["radius"])**2

    def _spawn_explosion(self, position, num_particles, colors):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                "pos": list(position),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "size": self.np_random.uniform(2, 6),
                "lifetime": self.np_random.integers(15, 30),
                "max_lifetime": 30,
                "color": random.choice(colors),
            })
    
    def _get_spawn_points(self, num_points, excluded_points):
        available_points = []
        for r in range(self.GRID_ROWS - 2): # Enemies only in top rows
            for c in range(self.GRID_COLS):
                if [c, r] not in excluded_points:
                    available_points.append([c, r])
        
        if num_points > len(available_points):
            return available_points # Return all if not enough
        
        chosen_indices = self.np_random.choice(len(available_points), num_points, replace=False)
        return [available_points[i] for i in chosen_indices]

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a pygame window.
    # The environment is designed for headless rendering (rgb_array),
    # but we can display the output for demonstration.
    
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Robot Arena")
        
        obs, info = env.reset()
        done = False
        
        while not done:
            # --- Map keyboard to MultiDiscrete action ---
            keys = pygame.key.get_pressed()
            
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Step environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # --- Render to screen ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # --- Event handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset() # Reset on 'r' key
    
    finally:
        pygame.quit()
        print(f"Final Score: {info['score']}")