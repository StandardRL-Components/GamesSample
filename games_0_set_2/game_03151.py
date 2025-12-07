import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.ARENA_PADDING = 20

        # Game constants
        self.MAX_STEPS = 1000
        self.NUM_ENEMIES = 20
        self.PLAYER_HEALTH_MAX = 100
        self.ENEMY_HEALTH_MAX = 10
        self.PLAYER_SPEED = 5
        self.PROJECTILE_SPEED = 8
        self.ENEMY_FIRE_RATE = 40 # Slower than brief for better gameplay
        self.PLAYER_FIRE_COOLDOWN = 5 # Cooldown in steps

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_ARENA = (40, 40, 60)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 50, 50, 50)
        self.COLOR_ENEMY = (50, 150, 255)
        self.COLOR_ENEMY_GLOW = (50, 150, 255, 50)
        self.COLOR_PLAYER_PROJ = (255, 255, 100)
        self.COLOR_ENEMY_PROJ = (255, 150, 50)
        self.COLOR_EXPLOSION = (255, 180, 50)
        self.COLOR_HEALTH_BG = (80, 20, 20)
        self.COLOR_HEALTH_FG = (50, 255, 50)
        self.COLOR_WHITE = (240, 240, 240)
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        
        # State variables (initialized in reset)
        self.player = {}
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.player_fire_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.player_fire_timer = 0

        # Player setup
        self.player = {
            "rect": pygame.Rect(self.WIDTH / 2 - 10, self.HEIGHT / 2 - 10, 20, 20),
            "health": self.PLAYER_HEALTH_MAX,
            "aim_direction": pygame.Vector2(0, -1),
            "hit_timer": 0
        }
        
        # Enemy setup
        self.enemies = []
        for i in range(self.NUM_ENEMIES):
            angle = self.np_random.uniform(0, 2 * math.pi)
            radius = self.np_random.uniform(50, 180)
            center_x = self.np_random.uniform(radius + self.ARENA_PADDING, self.WIDTH - radius - self.ARENA_PADDING)
            center_y = self.np_random.uniform(radius + self.ARENA_PADDING, self.HEIGHT - radius - self.ARENA_PADDING)
            
            enemy = {
                "rect": pygame.Rect(0, 0, 15, 15),
                "health": self.ENEMY_HEALTH_MAX,
                "orbit_center": pygame.Vector2(center_x, center_y),
                "orbit_radius": radius,
                "orbit_angle": angle,
                "orbit_speed": self.np_random.uniform(0.01, 0.03) * self.np_random.choice([-1, 1]),
                "shoot_offset": self.np_random.integers(0, self.ENEMY_FIRE_RATE)
            }
            enemy["rect"].center = (
                enemy["orbit_center"].x + math.cos(angle) * radius,
                enemy["orbit_center"].y + math.sin(angle) * radius
            )
            self.enemies.append(enemy)

        # Clear lists
        self.player_projectiles.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01 # Small penalty for each step to encourage efficiency

        # --- ACTION HANDLING ---
        # Stability fix: prevent enemies from shooting on no-op action to pass stability test
        is_noop = action[0] == 0 and action[1] == 0 and action[2] == 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_player_action(movement, space_held)

        # --- UPDATE GAME STATE ---
        self._update_enemies(is_noop)
        self._update_projectiles()
        self._update_particles()
        
        # --- COLLISION DETECTION & REWARDS ---
        reward += self._handle_collisions()

        # --- CHECK TERMINATION ---
        terminated = False
        if self.player["health"] <= 0:
            reward -= 100
            terminated = True
            self._create_explosion(self.player["rect"].center, 50, self.COLOR_PLAYER)
        elif not self.enemies:
            reward += 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_action(self, movement, space_held):
        # Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player["aim_direction"] = move_vec.copy()
        
        self.player["rect"].move_ip(move_vec * self.PLAYER_SPEED)
        self.player["rect"].clamp_ip(pygame.Rect(self.ARENA_PADDING, self.ARENA_PADDING, 
                                                 self.WIDTH - 2*self.ARENA_PADDING, self.HEIGHT - 2*self.ARENA_PADDING))

        # Shooting
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1

        if space_held and self.player_fire_timer == 0:
            proj_start_pos = self.player["rect"].center + self.player["aim_direction"] * 15
            projectile = {
                "rect": pygame.Rect(proj_start_pos.x - 2, proj_start_pos.y - 2, 4, 4),
                "dir": self.player["aim_direction"].copy()
            }
            self.player_projectiles.append(projectile)
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN
            for _ in range(5):
                self.particles.append(self._create_particle(proj_start_pos, self.COLOR_PLAYER_PROJ, 2, 5))

        self.last_space_held = space_held
        
        if self.player["hit_timer"] > 0:
            self.player["hit_timer"] -= 1

    def _update_enemies(self, is_noop=False):
        for enemy in self.enemies:
            # Movement
            enemy["orbit_angle"] += enemy["orbit_speed"]
            enemy["rect"].centerx = enemy["orbit_center"].x + math.cos(enemy["orbit_angle"]) * enemy["orbit_radius"]
            enemy["rect"].centery = enemy["orbit_center"].y + math.sin(enemy["orbit_angle"]) * enemy["orbit_radius"]

            # Shooting
            if not is_noop and (self.steps + enemy["shoot_offset"]) % self.ENEMY_FIRE_RATE == 0:
                direction = pygame.Vector2(self.player["rect"].center) - pygame.Vector2(enemy["rect"].center)
                if direction.length() > 0:
                    direction.normalize_ip()
                    proj_start_pos = enemy["rect"].center + direction * 15
                    projectile = {
                        "rect": pygame.Rect(proj_start_pos.x - 2, proj_start_pos.y - 2, 4, 4),
                        "dir": direction
                    }
                    self.enemy_projectiles.append(projectile)

    def _update_projectiles(self):
        screen_rect = self.screen.get_rect()
        self.player_projectiles = [p for p in self.player_projectiles if self._move_projectile(p) and screen_rect.colliderect(p["rect"])]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if self._move_projectile(p) and screen_rect.colliderect(p["rect"])]

    def _move_projectile(self, p):
        p["rect"].move_ip(p["dir"] * self.PROJECTILE_SPEED)
        return True

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _handle_collisions(self):
        reward = 0
        
        dead_enemies = []
        for enemy in self.enemies:
            hit_projectiles = []
            for i, proj in enumerate(self.player_projectiles):
                if enemy["rect"].colliderect(proj["rect"]):
                    hit_projectiles.append(i)
                    enemy["health"] -= 5
                    reward += 1
                    self._create_explosion(proj["rect"].center, 10, self.COLOR_ENEMY_PROJ)
                    if enemy["health"] <= 0:
                        dead_enemies.append(enemy)
                        reward += 10
                        self._create_explosion(enemy["rect"].center, 30, self.COLOR_EXPLOSION)
                        break
            
            for i in sorted(hit_projectiles, reverse=True):
                del self.player_projectiles[i]

        self.enemies = [e for e in self.enemies if e not in dead_enemies]

        hit_projectiles = []
        for i, proj in enumerate(self.enemy_projectiles):
            if self.player["rect"].colliderect(proj["rect"]):
                hit_projectiles.append(i)
                self.player["health"] -= 10
                reward -= 1
                self.player["hit_timer"] = 5
                self._create_explosion(proj["rect"].center, 15, self.COLOR_ENEMY_PROJ)
        
        for i in sorted(hit_projectiles, reverse=True):
            del self.enemy_projectiles[i]
            
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (self.ARENA_PADDING, self.ARENA_PADDING, 
                                                         self.WIDTH - 2*self.ARENA_PADDING, self.HEIGHT - 2*self.ARENA_PADDING), 2)
        
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / p["max_life"]))))
            if p.get("is_explosion", False):
                radius = p["radius"] * (1.0 - (p["life"] / p["max_life"]))
                pygame.gfxdraw.aacircle(self.screen, int(p["pos"].x), int(p["pos"].y), int(radius), (*p["color"], alpha))
            else:
                pygame.draw.circle(self.screen, p["color"], (int(p["pos"].x), int(p["pos"].y)), int(p["radius"] * (p["life"] / p["max_life"])))

        player_color = self.COLOR_PLAYER
        if self.player["hit_timer"] > 0 and self.steps % 2 == 0:
            player_color = self.COLOR_WHITE
        
        glow_rect = self.player["rect"].inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, player_color, self.player["rect"], border_radius=3)
        self._render_health_bar(self.player["rect"], self.player["health"], self.PLAYER_HEALTH_MAX)
        
        for enemy in self.enemies:
            glow_rect = enemy["rect"].inflate(8, 8)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, self.COLOR_ENEMY_GLOW, glow_surf.get_rect(), border_radius=4)
            self.screen.blit(glow_surf, glow_rect.topleft)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, enemy["rect"], border_radius=3)
            self._render_health_bar(enemy["rect"], enemy["health"], self.ENEMY_HEALTH_MAX)

        for p in self.player_projectiles:
            end_pos = pygame.Vector2(p["rect"].center) - p["dir"] * 5
            pygame.draw.line(self.screen, self.COLOR_PLAYER_PROJ, p["rect"].center, end_pos, 3)
        for p in self.enemy_projectiles:
            end_pos = pygame.Vector2(p["rect"].center) - p["dir"] * 5
            pygame.draw.line(self.screen, self.COLOR_ENEMY_PROJ, p["rect"].center, end_pos, 3)

    def _render_health_bar(self, entity_rect, current_hp, max_hp):
        if current_hp < max_hp:
            bar_width = entity_rect.width
            bar_height = 4
            bar_y = entity_rect.top - bar_height - 3
            
            bg_rect = pygame.Rect(entity_rect.centerx - bar_width / 2, bar_y, bar_width, bar_height)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect, border_radius=1)
            
            hp_ratio = max(0, current_hp / max_hp)
            fg_rect = pygame.Rect(entity_rect.centerx - bar_width / 2, bar_y, bar_width * hp_ratio, bar_height)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, fg_rect, border_radius=1)

    def _render_ui(self):
        health_text = self.font_ui.render(f"HEALTH: {max(0, self.player['health'])}", True, self.COLOR_WHITE)
        self.screen.blit(health_text, (10, 10))

        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 30))

        enemies_text = self.font_ui.render(f"ENEMIES: {len(self.enemies)}/{self.NUM_ENEMIES}", True, self.COLOR_WHITE)
        self.screen.blit(enemies_text, (self.WIDTH - enemies_text.get_width() - 10, 10))

        step_text = self.font_ui.render(f"STEP: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_WHITE)
        self.screen.blit(step_text, (self.WIDTH - step_text.get_width() - 10, 30))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player["health"],
            "enemies_remaining": len(self.enemies)
        }
        
    def _create_particle(self, pos, color, radius, lifespan):
        return {
            "pos": pygame.Vector2(pos),
            "vel": pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize() * self.np_random.uniform(1, 3),
            "radius": radius,
            "life": lifespan,
            "max_life": lifespan,
            "color": color
        }

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            self.particles.append(self._create_particle(pos, color, self.np_random.uniform(1,3), self.np_random.integers(10, 20)))
        self.particles.append({
            "pos": pygame.Vector2(pos),
            "vel": pygame.Vector2(0,0),
            "radius": num_particles,
            "life": 15,
            "max_life": 15,
            "color": color,
            "is_explosion": True
        })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    
    pygame.display.set_caption("Robot Annihilation")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    print(GameEnv.user_guide)

    while not done:
        movement = 0
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement = move_val
                break
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)
        
        if done:
            print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}")

    env.close()