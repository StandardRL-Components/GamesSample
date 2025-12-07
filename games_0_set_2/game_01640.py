
# Generated: 2025-08-27T17:47:51.070694
# Source Brief: brief_01640.md
# Brief Index: 1640

        
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
        "Controls: Arrow keys to move. Hold Space to fire. Avoid enemy fire and destroy 50 robots."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a powerful robot in a top-down neon arena, blasting waves of enemy robots to achieve ultimate robotic dominance."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 10000
    WIN_CONDITION_KILLS = 50

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_ARENA = (30, 35, 50)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 50)
    COLOR_PLAYER_PROJ = (255, 255, 0)
    COLOR_PLAYER_PROJ_GLOW = (255, 255, 0, 100)
    COLOR_ENEMY_PROJ = (255, 100, 0)
    COLOR_ENEMY_PROJ_GLOW = (255, 100, 0, 100)
    COLOR_HEALTH_POWERUP = (0, 255, 100)
    COLOR_DAMAGE_POWERUP = (255, 150, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    COLOR_HEALTH_BAR = (0, 200, 80)

    # Player
    PLAYER_RADIUS = 12
    PLAYER_SPEED = 4.0
    PLAYER_MAX_HEALTH = 100
    PLAYER_SHOOT_COOLDOWN = 6  # frames

    # Enemy
    ENEMY_RADIUS = 10
    ENEMY_BASE_SPEED = 1.0
    ENEMY_MAX_HEALTH = 30
    ENEMY_SHOOT_COOLDOWN = 60 # 2 seconds at 30fps

    # Projectile
    PROJ_RADIUS = 4
    PROJ_SPEED = 8.0
    PROJ_BASE_DAMAGE = 10

    # Powerups
    POWERUP_RADIUS = 8
    POWERUP_LIFETIME = 300 # 10 seconds
    POWERUP_SPAWN_CHANCE = 0.01

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
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 24)

        self.game_over_font = pygame.font.SysFont("monospace", 48, bold=True)

        self.render_mode = render_mode
        self.np_random = None

        self.reset()
        
        # This will run once and confirm the implementation is correct.
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player = {
            "pos": pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2),
            "health": self.PLAYER_MAX_HEALTH,
            "aim_dir": pygame.math.Vector2(0, -1),
            "shoot_cooldown": 0,
            "damage_mod": 1.0,
            "damage_mod_timer": 0,
            "kills": 0,
        }
        
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.powerups = []
        self.particles = []

        self.enemy_base_speed = self.ENEMY_BASE_SPEED
        self.max_enemies = 2
        
        for _ in range(self.max_enemies):
            self._spawn_enemy()

        self.previous_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self.get_info()
            
        step_reward = -0.001 # Small penalty for existing

        self._handle_input(action)
        self._update_player()
        step_reward += self._update_enemies()
        self._update_projectiles()
        self._update_powerups()
        self._update_particles()
        
        collision_reward = self._handle_collisions()
        step_reward += collision_reward

        self._spawn_entities()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                step_reward += 100
            else: # loss
                step_reward -= 100

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        move_vec = pygame.math.Vector2(0, 0)
        
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player["pos"] += move_vec * self.PLAYER_SPEED
            self.player["aim_dir"] = move_vec.copy()
        
        if space_held and self.player["shoot_cooldown"] <= 0:
            # sfx: player_shoot.wav
            self._spawn_projectile(self.player["pos"], self.player["aim_dir"], is_player=True)
            self.player["shoot_cooldown"] = self.PLAYER_SHOOT_COOLDOWN

    def _update_player(self):
        self.player["pos"].x = np.clip(self.player["pos"].x, self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player["pos"].y = np.clip(self.player["pos"].y, self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)
        
        if self.player["shoot_cooldown"] > 0:
            self.player["shoot_cooldown"] -= 1
        
        if self.player["damage_mod_timer"] > 0:
            self.player["damage_mod_timer"] -= 1
        else:
            self.player["damage_mod"] = 1.0

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies:
            direction_to_player = (self.player["pos"] - enemy["pos"]).normalize()
            enemy["pos"] += direction_to_player * self.enemy_base_speed
            
            enemy["shoot_cooldown"] -= 1
            if enemy["shoot_cooldown"] <= 0:
                # sfx: enemy_shoot.wav
                self._spawn_projectile(enemy["pos"], direction_to_player, is_player=False)
                enemy["shoot_cooldown"] = self.ENEMY_SHOOT_COOLDOWN + self.np_random.integers(-15, 15)
        return reward

    def _update_projectiles(self):
        self.player_projectiles = [p for p in self.player_projectiles if self._is_on_screen(p["pos"])]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if self._is_on_screen(p["pos"])]
        
        for p in self.player_projectiles + self.enemy_projectiles:
            p["pos"] += p["vel"]

    def _update_powerups(self):
        for p in self.powerups:
            p["timer"] -= 1
        self.powerups = [p for p in self.powerups if p["timer"] > 0]
        
    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if proj["pos"].distance_to(enemy["pos"]) < self.PROJ_RADIUS + self.ENEMY_RADIUS:
                    # sfx: enemy_hit.wav
                    damage = self.PROJ_BASE_DAMAGE * self.player["damage_mod"]
                    enemy["health"] -= damage
                    reward += 0.1
                    self.score += 1
                    self._create_hit_effect(proj["pos"], self.COLOR_ENEMY)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)

                    if enemy["health"] <= 0:
                        # sfx: enemy_explode.wav
                        reward += 1.0
                        self.score += 10
                        self.player["kills"] += 1
                        self._create_explosion(enemy["pos"], self.COLOR_ENEMY)
                        if enemy in self.enemies: self.enemies.remove(enemy)
                        self._update_difficulty()
                    break

        # Enemy projectiles vs Player
        for proj in self.enemy_projectiles[:]:
            if proj["pos"].distance_to(self.player["pos"]) < self.PROJ_RADIUS + self.PLAYER_RADIUS:
                # sfx: player_hit.wav
                self.player["health"] -= self.PROJ_BASE_DAMAGE
                reward -= 0.5
                self._create_hit_effect(proj["pos"], self.COLOR_PLAYER)
                if proj in self.enemy_projectiles: self.enemy_projectiles.remove(proj)
                break
        
        # Player vs Powerups
        for powerup in self.powerups[:]:
            if self.player["pos"].distance_to(powerup["pos"]) < self.PLAYER_RADIUS + self.POWERUP_RADIUS:
                # sfx: powerup_collect.wav
                if powerup["type"] == "health":
                    self.player["health"] = min(self.PLAYER_MAX_HEALTH, self.player["health"] + 50)
                    reward += 5.0
                    self.score += 25
                elif powerup["type"] == "damage":
                    self.player["damage_mod"] = 2.0
                    self.player["damage_mod_timer"] = 300 # 10 seconds
                    reward += 2.0
                    self.score += 15
                self.powerups.remove(powerup)
        
        return reward

    def _spawn_entities(self):
        if len(self.enemies) < self.max_enemies:
            self._spawn_enemy()
        
        if self.np_random.random() < self.POWERUP_SPAWN_CHANCE and len(self.powerups) < 3:
            self._spawn_powerup()

    def _check_termination(self):
        if self.player["health"] <= 0:
            self.game_over = True
            self.win = False
            return True
        if self.player["kills"] >= self.WIN_CONDITION_KILLS:
            self.game_over = True
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
        
    def _spawn_enemy(self):
        side = self.np_random.integers(4)
        if side == 0: # top
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -self.ENEMY_RADIUS)
        elif side == 1: # bottom
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ENEMY_RADIUS)
        elif side == 2: # left
            pos = pygame.math.Vector2(-self.ENEMY_RADIUS, self.np_random.uniform(0, self.HEIGHT))
        else: # right
            pos = pygame.math.Vector2(self.WIDTH + self.ENEMY_RADIUS, self.np_random.uniform(0, self.HEIGHT))
        
        self.enemies.append({
            "pos": pos,
            "health": self.ENEMY_MAX_HEALTH,
            "shoot_cooldown": self.ENEMY_SHOOT_COOLDOWN + self.np_random.integers(0, 60),
        })

    def _spawn_projectile(self, pos, direction, is_player):
        proj = {
            "pos": pos.copy(),
            "vel": direction.normalize() * self.PROJ_SPEED,
        }
        if is_player:
            self.player_projectiles.append(proj)
        else:
            self.enemy_projectiles.append(proj)

    def _spawn_powerup(self):
        ptype = self.np_random.choice(["health", "damage"])
        self.powerups.append({
            "pos": pygame.math.Vector2(
                self.np_random.uniform(50, self.WIDTH - 50),
                self.np_random.uniform(50, self.HEIGHT - 50)
            ),
            "type": ptype,
            "timer": self.POWERUP_LIFETIME
        })

    def _update_difficulty(self):
        # Enemy speed increases by 0.02 every 5 defeated robots.
        if self.player["kills"] > 0 and self.player["kills"] % 5 == 0:
            self.enemy_base_speed = min(self.ENEMY_BASE_SPEED + 0.02 * (self.player["kills"] // 5), self.PLAYER_SPEED - 0.5)

        # Max enemies increases by 1 every 10 defeated robots.
        if self.player["kills"] > 0 and self.player["kills"] % 10 == 0:
            self.max_enemies = min(2 + (self.player["kills"] // 10), 10)

    def _is_on_screen(self, pos):
        return 0 <= pos.x <= self.WIDTH and 0 <= pos.y <= self.HEIGHT

    def _create_explosion(self, pos, color):
        for _ in range(30):
            vel = pygame.math.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3))
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color,
                "radius": self.np_random.uniform(1, 5)
            })

    def _create_hit_effect(self, pos, color):
        for _ in range(5):
            vel = pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(5, 10),
                "color": color,
                "radius": self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (10, 10, self.WIDTH - 20, self.HEIGHT - 20), border_radius=5)
        
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Powerups
        for p in self.powerups:
            color = self.COLOR_HEALTH_POWERUP if p["type"] == "health" else self.COLOR_DAMAGE_POWERUP
            alpha = int(255 * (p["timer"] / self.POWERUP_LIFETIME))
            glow_color = (*color, int(100 * (p["timer"] / self.POWERUP_LIFETIME)))
            self._draw_glowing_circle(p["pos"], self.POWERUP_RADIUS, color, glow_color)
            symbol = "+" if p["type"] == "health" else "↑"
            text = self.font_small.render(symbol, True, self.COLOR_BG)
            self.screen.blit(text, text.get_rect(center=p["pos"]))
            
        # Projectiles
        for p in self.player_projectiles:
            self._draw_glowing_circle(p["pos"], self.PROJ_RADIUS, self.COLOR_PLAYER_PROJ, self.COLOR_PLAYER_PROJ_GLOW)
        for p in self.enemy_projectiles:
            self._draw_glowing_circle(p["pos"], self.PROJ_RADIUS, self.COLOR_ENEMY_PROJ, self.COLOR_ENEMY_PROJ_GLOW)

        # Enemies
        for e in self.enemies:
            self._draw_glowing_circle(e["pos"], self.ENEMY_RADIUS, self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW)
            # Health bar for enemy
            health_ratio = max(0, e["health"] / self.ENEMY_MAX_HEALTH)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (e["pos"].x - 10, e["pos"].y - 18, 20, 3))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (e["pos"].x - 10, e["pos"].y - 18, 20 * health_ratio, 3))

        # Player
        if self.player["health"] > 0:
            glow_color = self.COLOR_PLAYER_GLOW
            if self.player["damage_mod_timer"] > 0:
                # Flash orange when damage boost is active
                if (self.steps // 3) % 2 == 0:
                    glow_color = (*self.COLOR_DAMAGE_POWERUP, 150)

            self._draw_glowing_circle(self.player["pos"], self.PLAYER_RADIUS, self.COLOR_PLAYER, glow_color)
            turret_end = self.player["pos"] + self.player["aim_dir"] * self.PLAYER_RADIUS
            pygame.draw.line(self.screen, self.COLOR_TEXT, self.player["pos"], turret_end, 2)

        # Particles
        for p in self.particles:
            alpha = max(0, int(255 * (p["life"] / 30.0)))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color)

    def _draw_glowing_circle(self, pos, radius, color, glow_color):
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius + 3), glow_color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), color)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), color)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player["health"] / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (20, 20, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (20, 20, 200 * health_ratio, 20))
        health_text = self.font_small.render(f"HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (25, 22))

        # Score and Kills
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 15))
        
        kills_text = self.font_small.render(f"KILLS: {self.player['kills']} / {self.WIN_CONDITION_KILLS}", True, self.COLOR_TEXT)
        self.screen.blit(kills_text, (self.WIDTH - kills_text.get_width() - 20, 40))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            result_text_str = "VICTORY" if self.win else "GAME OVER"
            result_text = self.game_over_font.render(result_text_str, True, self.COLOR_PLAYER if self.win else self.COLOR_ENEMY)
            self.screen.blit(result_text, result_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20)))
            
            final_score_text = self.font_large.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            self.screen.blit(final_score_text, final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 30)))


    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "kills": self.player["kills"]}

    def close(self):
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

# Example usage to run and visualize the game
if __name__ == "__main__":
    import os
    # Set a dummy video driver to run headless if not on a desktop
    if "DISPLAY" not in os.environ:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Play ---
    # To play the game yourself, you need a display.
    # This code block will only work if you have a screen.
    try:
        pygame.display.set_caption("Robot Arena")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        
        obs, info = env.reset()
        done = False
        
        while not done:
            # --- Action Mapping for Human ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    env.reset()

        print(f"Game Over. Final Info: {info}")

    except pygame.error as e:
        print("Pygame display could not be initialized. Running in headless mode.")
        print("To play the game, run this script in an environment with a display.")
        # Simple random agent loop for headless validation
        obs, info = env.reset()
        for _ in range(2000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished. Final Info: {info}")
                obs, info = env.reset()

    env.close()