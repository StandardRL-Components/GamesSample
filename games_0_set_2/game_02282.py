
# Generated: 2025-08-27T19:53:42.839307
# Source Brief: brief_02282.md
# Brief Index: 2282

        
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


# Helper function for vector operations
def vec_to_angle(v):
    return math.atan2(v[1], v[0])

def angle_to_vec(angle):
    return pygame.Vector2(math.cos(angle), math.sin(angle))

# Helper function for drawing glowing circles
def draw_glow_circle(surface, color, center, radius, glow_strength):
    for i in range(glow_strength, 0, -1):
        alpha = 150 * (1 - i / glow_strength)
        pygame.gfxdraw.filled_circle(
            surface,
            int(center[0]),
            int(center[1]),
            int(radius + i),
            (*color, alpha)
        )
    pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius), color)
    pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), int(radius), color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold Shift to rotate your turret independently. Hold Space to fire."
    )

    game_description = (
        "Pilot a neon mech through a dark cityscape. Annihilate 15 enemy robots to win, but watch your health."
    )

    auto_advance = True

    # --- Constants ---
    # Sizes
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_WIDTH, WORLD_HEIGHT = 1800, 1200
    
    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_BULLET = (100, 200, 255)
    COLOR_ENEMY_PATROLLER = (255, 50, 50)
    COLOR_ENEMY_CHASER = (255, 120, 50)
    COLOR_ENEMY_SHOOTER = (255, 180, 50)
    COLOR_ENEMY_BULLET = (255, 255, 100)
    COLOR_EXPLOSION = [(255, 255, 255), (255, 200, 0), (255, 100, 0)]
    COLOR_BUILDING_OUTLINE = (50, 40, 80)
    COLOR_BUILDING_FILL = (30, 20, 50)
    
    # Game parameters
    PLAYER_MAX_HEALTH = 100
    PLAYER_SPEED = 4.0
    PLAYER_RADIUS = 12
    PLAYER_FIRE_COOLDOWN = 6  # frames
    PLAYER_TURRET_ROTATION_SPEED = 0.1  # radians per frame
    
    ENEMY_TARGET_COUNT = 15
    ENEMY_BASE_SPEED = 1.5
    ENEMY_BASE_FIRE_RATE = 0.02 # shots per frame
    
    BULLET_SPEED = 8.0
    BULLET_RADIUS = 3

    MAX_STEPS = 30 * 60 # 60 seconds at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.world_surface = pygame.Surface((self.WORLD_WIDTH, self.WORLD_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 36, bold=True)

        self.buildings = []
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = 0
        self.player_aim_angle = 0.0
        self.player_last_move_angle = 0.0
        self.player_fire_cooldown = 0
        
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.enemies_destroyed = 0
        self.game_over = False
        
        self.reset()
        
        # self.validate_implementation() # Optional self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.enemies_destroyed = 0
        self.game_over = False
        
        self._generate_world()
        
        self.player_pos = self._get_valid_spawn_pos(self.PLAYER_RADIUS)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_aim_angle = -math.pi / 2
        self.player_last_move_angle = -math.pi / 2
        self.player_fire_cooldown = 0
        
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.enemy_speed_multiplier = 1.0
        self.enemy_fire_rate_multiplier = 1.0
        
        for _ in range(self.ENEMY_TARGET_COUNT):
            self._spawn_enemy()
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        self._update_player(movement, space_held, shift_held)
        reward += self._update_enemies()
        reward += self._update_projectiles()
        self._update_particles()
        
        self.steps += 1
        
        # Check termination conditions
        terminated = False
        if self.player_health <= 0:
            terminated = True
            reward -= 100
            self._create_explosion(self.player_pos, self.PLAYER_RADIUS * 2, 50)
            # sfx: player_explosion
        elif self.enemies_destroyed >= self.ENEMY_TARGET_COUNT:
            terminated = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        if terminated:
            self.game_over = True
            
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement, space_held, shift_held):
        # --- Movement ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_last_move_angle = vec_to_angle(move_vec)
        
        # Collision with buildings
        new_pos = self.player_pos + move_vec * self.PLAYER_SPEED
        # Check X and Y movement separately to slide along walls
        next_rect_x = pygame.Rect(new_pos.x - self.PLAYER_RADIUS, self.player_pos.y - self.PLAYER_RADIUS, self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)
        if not self._check_building_collision(next_rect_x):
            self.player_pos.x = new_pos.x
        next_rect_y = pygame.Rect(self.player_pos.x - self.PLAYER_RADIUS, new_pos.y - self.PLAYER_RADIUS, self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)
        if not self._check_building_collision(next_rect_y):
            self.player_pos.y = new_pos.y

        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WORLD_WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.WORLD_HEIGHT)

        # --- Aiming ---
        if shift_held:
            self.player_aim_angle += self.PLAYER_TURRET_ROTATION_SPEED
        elif move_vec.length() > 0:
            self.player_aim_angle = self.player_last_move_angle

        # --- Firing ---
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
            
        if space_held and self.player_fire_cooldown <= 0:
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN
            direction = angle_to_vec(self.player_aim_angle)
            start_pos = self.player_pos + direction * (self.PLAYER_RADIUS + self.BULLET_RADIUS)
            self.projectiles.append({
                "pos": start_pos,
                "vel": direction * self.BULLET_SPEED,
                "owner": "player",
                "color": self.COLOR_PLAYER_BULLET,
                "radius": self.BULLET_RADIUS,
            })
            # sfx: player_shoot

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies:
            # --- Movement ---
            move_vec = pygame.Vector2(0, 0)
            if enemy["type"] == "patroller":
                enemy["pos"].x += enemy["vel"].x * self.enemy_speed_multiplier
                if enemy["pos"].x < enemy["patrol_min"] or enemy["pos"].x > enemy["patrol_max"]:
                    enemy["vel"].x *= -1
            elif enemy["type"] == "chaser":
                move_vec = self.player_pos - enemy["pos"]
                if move_vec.length() > 200: # Only chase when close
                    if move_vec.length() > 0:
                        move_vec.normalize_ip()
                        enemy["pos"] += move_vec * self.ENEMY_BASE_SPEED * self.enemy_speed_multiplier * 0.8
            # Chaser and Patroller collision with buildings
            if enemy["type"] != "shooter":
                enemy_rect = pygame.Rect(enemy["pos"].x - enemy["radius"], enemy["pos"].y - enemy["radius"], enemy["radius"]*2, enemy["radius"]*2)
                if self._check_building_collision(enemy_rect):
                    # Simple bounce back
                    enemy["pos"] -= move_vec * self.ENEMY_BASE_SPEED * self.enemy_speed_multiplier * 1.1

            # --- Firing ---
            enemy["fire_cooldown"] -= 1
            can_fire = enemy["fire_cooldown"] <= 0
            
            if can_fire and enemy["type"] != "patroller":
                target_vec = self.player_pos - enemy["pos"]
                if target_vec.length() < 300: # Firing range
                    enemy["fire_cooldown"] = self.np_random.integers(100, 150) / self.enemy_fire_rate_multiplier
                    direction = target_vec.normalize()
                    start_pos = enemy["pos"] + direction * (enemy["radius"] + self.BULLET_RADIUS)
                    self.projectiles.append({
                        "pos": start_pos,
                        "vel": direction * self.BULLET_SPEED * 0.8,
                        "owner": "enemy",
                        "color": self.COLOR_ENEMY_BULLET,
                        "radius": self.BULLET_RADIUS,
                    })
                    # sfx: enemy_shoot
        return reward

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for proj in self.projectiles:
            proj["pos"] += proj["vel"]
            
            # Check boundaries
            if not (0 < proj["pos"].x < self.WORLD_WIDTH and 0 < proj["pos"].y < self.WORLD_HEIGHT):
                continue

            # Check building collision
            proj_rect = pygame.Rect(proj["pos"].x - proj["radius"], proj["pos"].y - proj["radius"], proj["radius"]*2, proj["radius"]*2)
            if self._check_building_collision(proj_rect):
                self._create_explosion(proj["pos"], 5, 5)
                continue

            # Check entity collisions
            hit = False
            if proj["owner"] == "player":
                for enemy in self.enemies:
                    if proj["pos"].distance_to(enemy["pos"]) < enemy["radius"] + proj["radius"]:
                        enemy["health"] -= 25
                        reward += 0.1
                        hit = True
                        self._create_explosion(proj["pos"], 10, 10)
                        # sfx: enemy_hit
                        break
            elif proj["owner"] == "enemy":
                if proj["pos"].distance_to(self.player_pos) < self.PLAYER_RADIUS + proj["radius"]:
                    self.player_health -= 10
                    reward -= 0.1
                    hit = True
                    self._create_explosion(proj["pos"], 10, 10)
                    # sfx: player_hit
            
            if not hit:
                projectiles_to_keep.append(proj)
        
        self.projectiles = projectiles_to_keep

        # Check for destroyed enemies
        enemies_to_keep = []
        for enemy in self.enemies:
            if enemy["health"] > 0:
                enemies_to_keep.append(enemy)
            else:
                reward += 1
                self.enemies_destroyed += 1
                self._create_explosion(enemy["pos"], enemy["radius"] * 2.5, 30)
                # sfx: enemy_explosion
                if self.enemies_destroyed > 0 and self.enemies_destroyed % 5 == 0:
                    self.enemy_speed_multiplier += 0.1
                    self.enemy_fire_rate_multiplier += 0.2
        self.enemies = enemies_to_keep

        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] -= p["decay"]
    
    def _get_observation(self):
        # Camera follows player
        camera_x = self.player_pos.x - self.SCREEN_WIDTH / 2
        camera_y = self.player_pos.y - self.SCREEN_HEIGHT / 2
        
        # Blit world background
        self.screen.blit(self.world_surface, (-camera_x, -camera_y))
        
        # --- Render Game Elements ---
        
        # Render projectiles
        for proj in self.projectiles:
            screen_pos = proj["pos"] - pygame.Vector2(camera_x, camera_y)
            if self.screen.get_rect().collidepoint(screen_pos):
                draw_glow_circle(self.screen, proj["color"], screen_pos, proj["radius"], 5)

        # Render enemies
        for enemy in self.enemies:
            screen_pos = enemy["pos"] - pygame.Vector2(camera_x, camera_y)
            if self.screen.get_rect().collidepoint(screen_pos):
                draw_glow_circle(self.screen, enemy["color"], screen_pos, enemy["radius"], 10)

        # Render player
        player_screen_pos = self.player_pos - pygame.Vector2(camera_x, camera_y)
        draw_glow_circle(self.screen, self.COLOR_PLAYER, player_screen_pos, self.PLAYER_RADIUS, 15)
        
        # Render player turret
        turret_end = player_screen_pos + angle_to_vec(self.player_aim_angle) * (self.PLAYER_RADIUS + 5)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, player_screen_pos, turret_end, 3)

        # Render particles
        for p in self.particles:
            screen_pos = p["pos"] - pygame.Vector2(camera_x, camera_y)
            if p["radius"] > 0 and self.screen.get_rect().collidepoint(screen_pos):
                alpha = int(255 * (p["life"] / p["start_life"]))
                color = (*p["color"], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), int(p["radius"]), color)

        # Render UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Health bar
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        health_bar_width = 150
        health_bar_rect = pygame.Rect(10, 10, int(health_bar_width * health_ratio), 20)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, health_bar_rect)
        pygame.draw.rect(self.screen, (255,255,255), (10, 10, health_bar_width, 20), 1)
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Enemies remaining
        enemies_text = self.font_ui.render(f"TARGETS: {self.ENEMY_TARGET_COUNT - self.enemies_destroyed}", True, (255, 255, 255))
        self.screen.blit(enemies_text, (self.SCREEN_WIDTH / 2 - enemies_text.get_width() / 2, self.SCREEN_HEIGHT - 30))

        if self.game_over:
            if self.enemies_destroyed >= self.ENEMY_TARGET_COUNT:
                msg = "MISSION COMPLETE"
                color = self.COLOR_PLAYER
            else:
                msg = "MECH DESTROYED"
                color = self.COLOR_ENEMY_PATROLLER
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_big.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH / 2 - end_text.get_width() / 2, self.SCREEN_HEIGHT / 2 - end_text.get_height() / 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "enemies_destroyed": self.enemies_destroyed,
        }

    # --- Helper methods ---
    
    def _generate_world(self):
        self.world_surface.fill(self.COLOR_BG)
        self.buildings = []
        
        cell_size = 150
        for x in range(0, self.WORLD_WIDTH, cell_size):
            for y in range(0, self.WORLD_HEIGHT, cell_size):
                if self.np_random.random() < 0.4:
                    w = self.np_random.integers(40, cell_size - 20)
                    h = self.np_random.integers(40, cell_size - 20)
                    off_x = self.np_random.integers(10, cell_size - w - 10)
                    off_y = self.np_random.integers(10, cell_size - h - 10)
                    rect = pygame.Rect(x + off_x, y + off_y, w, h)
                    self.buildings.append(rect)
        
        for rect in self.buildings:
            pygame.draw.rect(self.world_surface, self.COLOR_BUILDING_FILL, rect)
            pygame.draw.rect(self.world_surface, self.COLOR_BUILDING_OUTLINE, rect, 2)
    
    def _get_valid_spawn_pos(self, radius):
        while True:
            pos = pygame.Vector2(
                self.np_random.integers(radius, self.WORLD_WIDTH - radius),
                self.np_random.integers(radius, self.WORLD_HEIGHT - radius)
            )
            rect = pygame.Rect(pos.x - radius, pos.y - radius, radius * 2, radius * 2)
            if not self._check_building_collision(rect):
                return pos

    def _check_building_collision(self, rect):
        return rect.collidelist(self.buildings) != -1

    def _spawn_enemy(self):
        enemy_type = self.np_random.choice(["patroller", "chaser", "shooter"], p=[0.4, 0.4, 0.2])
        radius = 10
        pos = self._get_valid_spawn_pos(radius)
        
        if enemy_type == "patroller":
            enemy = {
                "type": "patroller", "pos": pos, "radius": radius, "health": 50,
                "color": self.COLOR_ENEMY_PATROLLER, "vel": pygame.Vector2(self.ENEMY_BASE_SPEED, 0),
                "patrol_min": pos.x - 100, "patrol_max": pos.x + 100, "fire_cooldown": 100
            }
        elif enemy_type == "chaser":
            enemy = {
                "type": "chaser", "pos": pos, "radius": radius, "health": 75,
                "color": self.COLOR_ENEMY_CHASER, "fire_cooldown": 120
            }
        else: # shooter
            enemy = {
                "type": "shooter", "pos": pos, "radius": 14, "health": 100,
                "color": self.COLOR_ENEMY_SHOOTER, "fire_cooldown": 80
            }
        self.enemies.append(enemy)

    def _create_explosion(self, pos, size, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * (size / 10) + 1
            vel = angle_to_vec(angle) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": life,
                "start_life": life,
                "radius": self.np_random.random() * (size / 8) + 1,
                "decay": 0.1,
                "color": random.choice(self.COLOR_EXPLOSION)
            })

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Robo Rampage")
    clock = pygame.time.Clock()
    
    running = True
    total_score = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
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
        total_score += reward
        
        # Convert observation back to a Pygame surface for display
        # The observation is (H, W, C), Pygame needs (W, H, C) for surfarray
        # The env returns (H, W, C), but its internal screen is (W, H)
        # So we need to transpose it back for displaying
        display_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(display_surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_score = 0
        
        clock.tick(30) # Match the auto_advance rate
        
    pygame.quit()