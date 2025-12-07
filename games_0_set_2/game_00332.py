import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move and aim. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a tank in a top-down arena. Blast 15 enemy tanks to win, but watch out for their return fire!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500  # Increased for more play time
        self.NUM_ENEMIES = 15

        # Player constants
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 4
        self.PLAYER_HEALTH_MAX = 100
        self.PLAYER_FIRE_COOLDOWN = 8  # frames

        # Enemy constants
        self.ENEMY_SIZE = 20
        self.ENEMY_SPEED = 1
        self.ENEMY_HEALTH_MAX = 20
        self.ENEMY_FIRE_COOLDOWN = 45 # Slower fire rate
        self.ENEMY_PATROL_CHANGE_INTERVAL = 90 # frames

        # Projectile constants
        self.PROJECTILE_RADIUS = 4
        self.PROJECTILE_SPEED = 8
        self.PROJECTILE_DAMAGE = 20

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_PLAYER = (50, 200, 50)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_PROJECTILE_PLAYER = (150, 255, 150)
        self.COLOR_PROJECTILE_ENEMY = (255, 150, 150)
        self.COLOR_EXPLOSION = (255, 165, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_GREEN = (0, 255, 0)
        self.COLOR_HEALTH_RED = (255, 0, 0)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Initialize state variables
        self.player = {}
        self.enemies = []
        self.projectiles = []
        self.explosions = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""

        # This ensures that the first call to reset() has a seed
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""

        # Player state
        self.player = {
            "rect": pygame.Rect(self.WIDTH // 2 - self.PLAYER_SIZE // 2, self.HEIGHT // 2 - self.PLAYER_SIZE // 2, self.PLAYER_SIZE, self.PLAYER_SIZE),
            "health": self.PLAYER_HEALTH_MAX,
            "angle": -math.pi / 2, # Start facing up
            "fire_cooldown": 0,
        }

        # Enemy state
        self.enemies = []
        for _ in range(self.NUM_ENEMIES):
            while True:
                x = self.np_random.integers(self.ENEMY_SIZE, self.WIDTH - self.ENEMY_SIZE)
                y = self.np_random.integers(self.ENEMY_SIZE, self.HEIGHT - self.ENEMY_SIZE)
                new_enemy_rect = pygame.Rect(x, y, self.ENEMY_SIZE, self.ENEMY_SIZE)
                # Ensure enemies don't spawn on the player
                if not new_enemy_rect.colliderect(self.player["rect"].inflate(50, 50)):
                    break
            
            patrol_dir = (self.np_random.choice([-1, 0, 1]), self.np_random.choice([-1, 0, 1]))
            if patrol_dir == (0,0): patrol_dir = (1,0) # Ensure it moves

            self.enemies.append({
                "rect": new_enemy_rect,
                "health": self.ENEMY_HEALTH_MAX,
                "angle": 0,
                "fire_cooldown": self.np_random.integers(self.ENEMY_FIRE_COOLDOWN, self.ENEMY_FIRE_COOLDOWN * 2),
                "patrol_dir": patrol_dir,
                "patrol_timer": self.ENEMY_PATROL_CHANGE_INTERVAL,
            })

        self.projectiles = []
        self.explosions = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = -0.01 # Small penalty for each step to encourage efficiency

        if not self.game_over:
            self.steps += 1
            
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            
            self._handle_input(movement, space_held)
            reward += self._update_enemies()
            reward += self._update_projectiles()
            self._update_explosions()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if (terminated or truncated) and not self.game_over:
            self.game_over = True
            if self.player['health'] <= 0:
                reward -= 50
                self.game_outcome = "DEFEAT"
            elif not self.enemies:
                reward += 100
                self.game_outcome = "VICTORY!"
            else: # Max steps reached
                self.game_outcome = "TIME UP"
        
        # In Gymnasium, `terminated` and `truncated` are separate flags.
        # `terminated` is for terminal states (win/loss). `truncated` is for time limits.
        is_terminated = self.player["health"] <= 0 or not self.enemies

        return (
            self._get_observation(),
            reward,
            is_terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement and aiming
        dx, dy = 0, 0
        if movement == 1: # Up
            dy = -1
            self.player["angle"] = -math.pi / 2
        elif movement == 2: # Down
            dy = 1
            self.player["angle"] = math.pi / 2
        elif movement == 3: # Left
            dx = -1
            self.player["angle"] = math.pi
        elif movement == 4: # Right
            dx = 1
            self.player["angle"] = 0
        
        self.player["rect"].x += dx * self.PLAYER_SPEED
        self.player["rect"].y += dy * self.PLAYER_SPEED

        # Boundary checks
        self.player["rect"].left = max(0, self.player["rect"].left)
        self.player["rect"].right = min(self.WIDTH, self.player["rect"].right)
        self.player["rect"].top = max(0, self.player["rect"].top)
        self.player["rect"].bottom = min(self.HEIGHT, self.player["rect"].bottom)

        # Firing
        if self.player["fire_cooldown"] > 0:
            self.player["fire_cooldown"] -= 1
        
        if space_held and self.player["fire_cooldown"] == 0:
            # sfx: player_shoot.wav
            self.player["fire_cooldown"] = self.PLAYER_FIRE_COOLDOWN
            start_pos = self.player["rect"].center
            vel_x = math.cos(self.player["angle"]) * self.PROJECTILE_SPEED
            vel_y = math.sin(self.player["angle"]) * self.PROJECTILE_SPEED
            self.projectiles.append({
                "rect": pygame.Rect(start_pos[0] - self.PROJECTILE_RADIUS, start_pos[1] - self.PROJECTILE_RADIUS, self.PROJECTILE_RADIUS*2, self.PROJECTILE_RADIUS*2),
                "vel": (vel_x, vel_y),
                "owner": "player"
            })

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies:
            # Movement
            enemy["patrol_timer"] -= 1
            if enemy["patrol_timer"] <= 0:
                enemy["patrol_timer"] = self.ENEMY_PATROL_CHANGE_INTERVAL
                patrol_dir = (self.np_random.choice([-1, 0, 1]), self.np_random.choice([-1, 0, 1]))
                if patrol_dir == (0,0): patrol_dir = (1,0)
                enemy["patrol_dir"] = patrol_dir

            enemy["rect"].x += enemy["patrol_dir"][0] * self.ENEMY_SPEED
            enemy["rect"].y += enemy["patrol_dir"][1] * self.ENEMY_SPEED
            
            # Boundary checks for enemies
            if enemy["rect"].left < 0 or enemy["rect"].right > self.WIDTH:
                enemy["patrol_dir"] = (-enemy["patrol_dir"][0], enemy["patrol_dir"][1])
            if enemy["rect"].top < 0 or enemy["rect"].bottom > self.HEIGHT:
                enemy["patrol_dir"] = (enemy["patrol_dir"][0], -enemy["patrol_dir"][1])

            enemy["rect"].left = max(0, enemy["rect"].left)
            enemy["rect"].right = min(self.WIDTH, enemy["rect"].right)
            enemy["rect"].top = max(0, enemy["rect"].top)
            enemy["rect"].bottom = min(self.HEIGHT, enemy["rect"].bottom)

            # Aiming
            dx = self.player["rect"].centerx - enemy["rect"].centerx
            dy = self.player["rect"].centery - enemy["rect"].centery
            enemy["angle"] = math.atan2(dy, dx)

            # Firing
            if enemy["fire_cooldown"] > 0:
                enemy["fire_cooldown"] -= 1
            elif enemy["fire_cooldown"] == 0:
                # sfx: enemy_shoot.wav
                enemy["fire_cooldown"] = self.ENEMY_FIRE_COOLDOWN
                start_pos = enemy["rect"].center
                vel_x = math.cos(enemy["angle"]) * self.PROJECTILE_SPEED
                vel_y = math.sin(enemy["angle"]) * self.PROJECTILE_SPEED
                self.projectiles.append({
                    "rect": pygame.Rect(start_pos[0] - self.PROJECTILE_RADIUS, start_pos[1] - self.PROJECTILE_RADIUS, self.PROJECTILE_RADIUS*2, self.PROJECTILE_RADIUS*2),
                    "vel": (vel_x, vel_y),
                    "owner": "enemy"
                })
        return reward

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for proj in self.projectiles:
            proj["rect"].x += proj["vel"][0]
            proj["rect"].y += proj["vel"][1]

            hit = False
            # Check for collision with walls
            if not (0 < proj["rect"].centerx < self.WIDTH and 0 < proj["rect"].centery < self.HEIGHT):
                hit = True

            # Check for collision with tanks
            if not hit:
                if proj["owner"] == "player":
                    for enemy in self.enemies[:]: # Iterate on a copy
                        if enemy["rect"].colliderect(proj["rect"]):
                            # sfx: enemy_hit.wav
                            enemy["health"] -= self.PROJECTILE_DAMAGE
                            reward += 0.1 # Reward for hitting
                            self._create_explosion(proj["rect"].center, self.PROJECTILE_RADIUS)
                            hit = True
                            if enemy["health"] <= 0:
                                # sfx: enemy_explode.wav
                                reward += 10 # Reward for destroying
                                self.score += 100
                                self._create_explosion(enemy["rect"].center, self.ENEMY_SIZE)
                                self.enemies.remove(enemy)
                            break # Projectile is consumed
                elif proj["owner"] == "enemy":
                    if self.player["rect"].colliderect(proj["rect"]):
                        # sfx: player_hit.wav
                        self.player["health"] -= self.PROJECTILE_DAMAGE
                        reward -= 0.2 # Penalty for being hit
                        self._create_explosion(proj["rect"].center, self.PROJECTILE_RADIUS)
                        hit = True

            if not hit:
                projectiles_to_keep.append(proj)
        
        self.projectiles = projectiles_to_keep
        return reward

    def _create_explosion(self, pos, start_radius):
        self.explosions.append({"pos": pos, "radius": start_radius, "max_radius": start_radius * 2.5, "life": 10})

    def _update_explosions(self):
        for explosion in self.explosions[:]:
            explosion["life"] -= 1
            prog = 1 - (explosion["life"] / 10)
            explosion["radius"] = explosion["max_radius"] * prog
            if explosion["life"] <= 0:
                self.explosions.remove(explosion)

    def _check_termination(self):
        if self.player["health"] <= 0:
            return True
        if not self.enemies:
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
        # Draw all game elements
        for proj in self.projectiles:
            color = self.COLOR_PROJECTILE_PLAYER if proj["owner"] == "player" else self.COLOR_PROJECTILE_ENEMY
            pygame.gfxdraw.filled_circle(self.screen, int(proj["rect"].centerx), int(proj["rect"].centery), self.PROJECTILE_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, int(proj["rect"].centerx), int(proj["rect"].centery), self.PROJECTILE_RADIUS, color)

        for enemy in self.enemies:
            self._draw_tank(enemy["rect"], enemy["angle"], self.COLOR_ENEMY)
            self._draw_health_bar(enemy["rect"], enemy["health"], self.ENEMY_HEALTH_MAX, self.COLOR_HEALTH_RED)

        if self.player["health"] > 0:
            self._draw_tank(self.player["rect"], self.player["angle"], self.COLOR_PLAYER)

        for explosion in self.explosions:
            alpha = int(255 * (explosion['life'] / 10))
            color = (*self.COLOR_EXPLOSION, alpha)
            
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, int(explosion["pos"][0]), int(explosion["pos"][1]), int(explosion["radius"]), color)
            self.screen.blit(s, (0,0))
    
    def _draw_tank(self, rect, angle, color):
        pygame.draw.rect(self.screen, color, rect)
        turret_len = self.PLAYER_SIZE * 0.8
        start_pos = rect.center
        end_pos = (
            start_pos[0] + turret_len * math.cos(angle),
            start_pos[1] + turret_len * math.sin(angle)
        )
        pygame.draw.line(self.screen, self.COLOR_WALL, start_pos, end_pos, 3)

    def _draw_health_bar(self, rect, current_health, max_health, color):
        if current_health < max_health:
            bar_width = rect.width
            bar_height = 5
            bar_x = rect.x
            bar_y = rect.y - bar_height - 3
            
            health_percent = max(0, current_health / max_health)
            
            pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_width * health_percent, bar_height))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Enemies remaining
        enemies_text = self.font_ui.render(f"ENEMIES: {len(self.enemies)}/{self.NUM_ENEMIES}", True, self.COLOR_TEXT)
        self.screen.blit(enemies_text, (self.WIDTH - enemies_text.get_width() - 10, 10))

        # Player Health Bar
        health_bar_width = 200
        health_bar_height = 20
        health_bar_x = (self.WIDTH - health_bar_width) // 2
        health_bar_y = self.HEIGHT - health_bar_height - 10
        health_percent = max(0, self.player["health"] / self.PLAYER_HEALTH_MAX)
        
        pygame.draw.rect(self.screen, (50,50,50), (health_bar_x, health_bar_y, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (health_bar_x, health_bar_y, health_bar_width * health_percent, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (health_bar_x, health_bar_y, health_bar_width, health_bar_height), 2)

        if self.game_over:
            outcome_color = self.COLOR_HEALTH_GREEN if self.game_outcome == "VICTORY!" else self.COLOR_HEALTH_RED
            if self.game_outcome == "TIME UP":
                outcome_color = self.COLOR_TEXT
            game_over_text = self.font_game_over.render(self.game_outcome, True, outcome_color)
            text_rect = game_over_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(game_over_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player.get("health", 0),
            "enemies_remaining": len(self.enemies),
        }

    def close(self):
        pygame.quit()
    
if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=random.randint(0, 10000))
    
    # Setup Pygame window for human play
    # This is required to get key presses.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.display.init()
    pygame.display.set_caption("Tank Arena")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    truncated = False
    
    print(env.user_guide)
    print(env.game_description)

    while not (terminated or truncated):
        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True # End the loop
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset(seed=random.randint(0, 10000))
                terminated = False
                truncated = False
    
    print(f"Game Over. Final Info: {info}")
    env.close()