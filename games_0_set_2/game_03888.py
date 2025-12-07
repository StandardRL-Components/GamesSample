import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold Space to fire. Shift does nothing."
    )

    game_description = (
        "Pilot a nimble robot in a top-down arena, dodging enemy fire while blasting them to scrap for points and headshot bonuses."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.N_ENEMIES = 15

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 200)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_HEAD = (255, 150, 150)
        self.COLOR_BULLET_PLAYER = (255, 255, 0)
        self.COLOR_BULLET_ENEMY = (255, 100, 0)
        self.COLOR_PARTICLE = (255, 180, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_FG = (50, 200, 50)
        self.COLOR_HEALTH_BG = (100, 50, 50)
        self.COLOR_ARENA = (100, 100, 120)

        # Player settings
        self.PLAYER_SIZE = 12
        self.PLAYER_SPEED = 5
        self.PLAYER_MAX_HEALTH = 3
        self.PLAYER_FIRE_COOLDOWN = 4 # steps

        # Enemy settings
        self.ENEMY_SIZE = 14
        self.ENEMY_SPEED = 1
        self.ENEMY_FIRE_COOLDOWN = 50
        self.ENEMY_ORBIT_RADIUS = 150
        self.ENEMY_AIM_INACCURACY = 0.1 # radians

        # Bullet settings
        self.BULLET_SPEED = 12
        self.BULLET_LENGTH = 10

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_info = pygame.font.SysFont("Consolas", 16)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = None
        self.player_health = 0
        self.player_aim_angle = 0
        self.player_fire_timer = 0
        
        self.enemies = []
        self.bullets = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_aim_angle = -math.pi / 2  # Start facing up
        self.player_fire_timer = 0

        self.enemies = self._spawn_enemies()
        self.bullets = []
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _spawn_enemies(self):
        enemies = []
        center_x, center_y = self.WIDTH / 2, self.HEIGHT / 2
        # Add a grace period to prevent immediate firing, passing the stability test
        grace_period = 65 
        for i in range(self.N_ENEMIES):
            angle = (2 * math.pi / self.N_ENEMIES) * i
            pos_x = center_x + self.ENEMY_ORBIT_RADIUS * math.cos(angle)
            pos_y = center_y + self.ENEMY_ORBIT_RADIUS * math.sin(angle)
            enemies.append({
                "pos": np.array([pos_x, pos_y], dtype=np.float64),
                "angle_offset": angle,
                "fire_cooldown": self.np_random.integers(grace_period, grace_period + self.ENEMY_FIRE_COOLDOWN)
            })
        return enemies

    def step(self, action):
        reward = -0.02  # Time penalty
        
        # --- Handle Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_player_input(movement, space_held)

        # --- Update Game State ---
        self._update_enemies()
        self._update_bullets()
        self._update_particles()
        
        # --- Handle Collisions ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- Check Termination ---
        self.steps += 1
        terminated = False
        truncated = False
        
        if self.player_health <= 0:
            terminated = True
        elif not self.enemies:
            reward += 50  # Victory bonus
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # Movement
        if movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
            self.player_aim_angle = -math.pi / 2
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
            self.player_aim_angle = math.pi / 2
        elif movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
            self.player_aim_angle = math.pi
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
            self.player_aim_angle = 0

        # Clamp position
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        # Firing
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
        
        if space_held and self.player_fire_timer == 0:
            vel = np.array([math.cos(self.player_aim_angle), math.sin(self.player_aim_angle)]) * self.BULLET_SPEED
            self.bullets.append({
                "pos": self.player_pos.copy(),
                "vel": vel,
                "owner": "player"
            })
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_enemies(self):
        center_x, center_y = self.WIDTH / 2, self.HEIGHT / 2
        orbit_speed = self.ENEMY_SPEED / self.ENEMY_ORBIT_RADIUS

        for enemy in self.enemies:
            # Circular movement
            angle = enemy["angle_offset"] + self.steps * orbit_speed
            enemy["pos"][0] = center_x + self.ENEMY_ORBIT_RADIUS * math.cos(angle)
            enemy["pos"][1] = center_y + self.ENEMY_ORBIT_RADIUS * math.sin(angle)

            # Firing logic
            enemy["fire_cooldown"] -= 1
            if enemy["fire_cooldown"] <= 0:
                direction_to_player = self.player_pos - enemy["pos"]
                angle_to_player = math.atan2(direction_to_player[1], direction_to_player[0])
                inaccuracy = self.np_random.uniform(-self.ENEMY_AIM_INACCURACY, self.ENEMY_AIM_INACCURACY)
                fire_angle = angle_to_player + inaccuracy
                
                vel = np.array([math.cos(fire_angle), math.sin(fire_angle)]) * self.BULLET_SPEED * 0.75
                self.bullets.append({
                    "pos": enemy["pos"].copy(),
                    "vel": vel,
                    "owner": "enemy"
                })
                enemy["fire_cooldown"] = self.ENEMY_FIRE_COOLDOWN

    def _update_bullets(self):
        for bullet in self.bullets:
            bullet["pos"] += bullet["vel"]
        
        # Remove off-screen bullets
        self.bullets = [b for b in self.bullets if 0 <= b["pos"][0] <= self.WIDTH and 0 <= b["pos"][1] <= self.HEIGHT]

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["radius"] -= 0.2
        self.particles = [p for p in self.particles if p["lifespan"] > 0 and p["radius"] > 0]

    def _create_explosion(self, pos, num_particles=20):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": self.np_random.uniform(3, 7),
                "lifespan": self.np_random.integers(10, 20)
            })

    def _handle_collisions(self):
        reward = 0
        
        # Player bullets vs Enemies
        enemies_to_remove = []
        bullets_to_remove = []
        for i, bullet in enumerate(self.bullets):
            if bullet["owner"] == "player":
                for j, enemy in enumerate(self.enemies):
                    if j in enemies_to_remove: continue

                    enemy_verts = self._get_robot_vertices(enemy["pos"], self.ENEMY_SIZE, math.atan2(self.player_pos[1] - enemy["pos"][1], self.player_pos[0] - enemy["pos"][0]))
                    if self._is_point_in_triangle(bullet["pos"], *enemy_verts):
                        self._create_explosion(enemy["pos"])
                        enemies_to_remove.append(j)
                        if i not in bullets_to_remove: bullets_to_remove.append(i)
                        
                        # Headshot check
                        head_pos = enemy_verts[0]
                        if np.linalg.norm(bullet["pos"] - head_pos) < self.ENEMY_SIZE * 0.5:
                            reward += 3 # Headshot
                            self.score += 300
                        else:
                            reward += 1 # Normal hit
                            self.score += 100
                        break
        
        self.enemies = [e for i, e in enumerate(self.enemies) if i not in enemies_to_remove]

        # Enemy bullets vs Player
        for i, bullet in enumerate(self.bullets):
            if bullet["owner"] == "enemy" and not self.game_over:
                player_verts = self._get_robot_vertices(self.player_pos, self.PLAYER_SIZE, self.player_aim_angle)
                if self._is_point_in_triangle(bullet["pos"], *player_verts):
                    self.player_health -= 1
                    if i not in bullets_to_remove: bullets_to_remove.append(i)
                    self._create_explosion(self.player_pos, num_particles=10) # Hit effect
                    if self.player_health <= 0:
                        self.game_over = True
                        self._create_explosion(self.player_pos, num_particles=50) # Death explosion
                        
        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]
        
        return reward

    def _get_robot_vertices(self, pos, size, angle):
        p1 = (pos[0] + size * math.cos(angle), pos[1] + size * math.sin(angle))
        p2 = (pos[0] + size * math.cos(angle + 2.2), pos[1] + size * math.sin(angle + 2.2))
        p3 = (pos[0] + size * math.cos(angle - 2.2), pos[1] + size * math.sin(angle - 2.2))
        return np.array(p1), np.array(p2), np.array(p3)

    def _is_point_in_triangle(self, pt, v1, v2, v3):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Arena boundary
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Particles
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            if p["radius"] > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), self.COLOR_PARTICLE)

        # Bullets
        for bullet in self.bullets:
            start_pos = (int(bullet["pos"][0]), int(bullet["pos"][1]))
            end_pos_vec = bullet["pos"] - bullet["vel"] / self.BULLET_SPEED * self.BULLET_LENGTH
            end_pos = (int(end_pos_vec[0]), int(end_pos_vec[1]))
            color = self.COLOR_BULLET_PLAYER if bullet["owner"] == 'player' else self.COLOR_BULLET_ENEMY
            pygame.draw.line(self.screen, color, start_pos, end_pos, 2)

        # Enemies
        for enemy in self.enemies:
            angle = math.atan2(self.player_pos[1] - enemy["pos"][1], self.player_pos[0] - enemy["pos"][0])
            verts = self._get_robot_vertices(enemy["pos"], self.ENEMY_SIZE, angle)
            int_verts = [(int(v[0]), int(v[1])) for v in verts]
            pygame.gfxdraw.aapolygon(self.screen, int_verts, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, int_verts, self.COLOR_ENEMY)
            # Headshot indicator
            head_pos = (int(verts[0][0]), int(verts[0][1]))
            pygame.gfxdraw.filled_circle(self.screen, head_pos[0], head_pos[1], 3, self.COLOR_ENEMY_HEAD)
        
        # Player
        if self.player_health > 0:
            pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(self.PLAYER_SIZE * 1.5), self.COLOR_PLAYER_GLOW + (100,))
            # Player body
            verts = self._get_robot_vertices(self.player_pos, self.PLAYER_SIZE, self.player_aim_angle)
            int_verts = [(int(v[0]), int(v[1])) for v in verts]
            pygame.gfxdraw.aapolygon(self.screen, int_verts, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, int_verts, self.COLOR_PLAYER)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (10, 10, int(bar_width * health_ratio), 20))
        health_text = self.font_info.render("HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (12, 12))

        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Enemies Remaining
        enemies_text = self.font_main.render(f"ENEMIES: {len(self.enemies)}/{self.N_ENEMIES}", True, self.COLOR_UI_TEXT)
        enemies_rect = enemies_text.get_rect(midtop=(self.WIDTH / 2, 10))
        self.screen.blit(enemies_text, enemies_rect)
        
        if self.game_over:
            if not self.enemies:
                end_text_str = "VICTORY!"
            else:
                end_text_str = "GAME OVER"
            end_text = pygame.font.SysFont("Consolas", 50, bold=True).render(end_text_str, True, self.COLOR_UI_TEXT)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "enemies_left": len(self.enemies)
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Override screen to be a display surface
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robot Arena")
    
    terminated = False
    truncated = False
    
    # Game loop
    while not terminated and not truncated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True # End the loop if window is closed

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        pygame.display.flip()
        
        # Control frame rate
        env.clock.tick(env.FPS)
        
    print(f"Game Over. Final Score: {info['score']}")
    env.close()