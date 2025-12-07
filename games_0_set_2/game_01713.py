
# Generated: 2025-08-28T02:27:15.386611
# Source Brief: brief_01713.md
# Brief Index: 1713

        
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

    user_guide = (
        "Controls: Arrow keys to move. Hold space to fire. Collect green power-ups for an advantage."
    )

    game_description = (
        "Pilot a powerful robot in a top-down neon arena. Blast waves of enemy robots to achieve total domination and the highest score."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.game_font = pygame.font.SysFont("monospace", 20, bold=True)
        self.game_font_small = pygame.font.SysFont("monospace", 16, bold=True)

        # Game constants
        self.ARENA_MARGIN = 20
        self.MAX_STEPS = 5000
        self.TOTAL_ENEMIES = 20

        # Colors
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_GRID = (30, 20, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 75, 128)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (128, 25, 25)
        self.COLOR_PLAYER_BULLET = (100, 200, 255)
        self.COLOR_ENEMY_BULLET = (255, 150, 100)
        self.COLOR_POWERUP_SHIELD = (0, 255, 150)
        self.COLOR_POWERUP_FIRERATE = (255, 255, 0)
        self.COLOR_EXPLOSION = [(255, 200, 0), (255, 100, 0), (200, 50, 0)]
        self.COLOR_WHITE = (255, 255, 255)
        
        # State variables - initialized in reset()
        self.player = {}
        self.enemies = []
        self.projectiles = []
        self.powerups = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.enemies_defeated = 0
        self.game_over = False
        self.np_random = None

        # Call reset to initialize the state
        self.reset()
        
        # Final validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.enemies_defeated = 0
        self.game_over = False
        
        self.player = {
            "pos": pygame.math.Vector2(self.screen_width / 2, self.screen_height / 2),
            "size": 12,
            "speed": 4,
            "health": 100,
            "max_health": 100,
            "aim_vector": pygame.math.Vector2(0, -1),
            "fire_cooldown": 0,
            "fire_rate": 8, # frames between shots
            "hit_timer": 0,
            "shield_timer": 0,
            "fast_fire_timer": 0,
        }
        
        self.enemies = []
        self._spawn_enemies(self.TOTAL_ENEMIES)
        
        self.projectiles = []
        self.powerups = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        
        # --- 1. Handle Player Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        player_velocity = pygame.math.Vector2(0, 0)
        moved = False
        if movement == 1: player_velocity.y = -1; moved = True
        elif movement == 2: player_velocity.y = 1; moved = True
        elif movement == 3: player_velocity.x = -1; moved = True
        elif movement == 4: player_velocity.x = 1; moved = True
        
        if moved:
            player_velocity.normalize_ip()
            self.player["aim_vector"] = player_velocity.copy()

        if space_held and self.player["fire_cooldown"] <= 0:
            # sfx: player_shoot.wav
            self._fire_projectile(self.player["pos"], self.player["aim_vector"], "player")
            fire_rate = self.player["fire_rate"] / 2 if self.player["fast_fire_timer"] > 0 else self.player["fire_rate"]
            self.player["fire_cooldown"] = fire_rate
            reward -= 0.01 # Small cost for firing

        # --- 2. Update Game State ---
        self._update_player(player_velocity)
        self._update_enemies()
        self._update_projectiles()
        self._update_powerups()
        self._update_particles()
        
        # --- 3. Handle Collisions & Events ---
        reward += self._handle_collisions()
        reward += self._handle_powerup_collection()
        
        # --- 4. Check Termination ---
        terminated = False
        if self.player["health"] <= 0:
            # sfx: player_explosion.wav
            self._create_explosion(self.player["pos"], 40, 2.0)
            reward = -100.0
            terminated = True
            self.game_over = True
        elif len(self.enemies) == 0:
            reward = 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_enemies(self, count):
        for _ in range(count):
            side = self.np_random.integers(4)
            if side == 0: x, y = self.np_random.uniform(0, self.screen_width), -30
            elif side == 1: x, y = self.screen_width + 30, self.np_random.uniform(0, self.screen_height)
            elif side == 2: x, y = self.np_random.uniform(0, self.screen_width), self.screen_height + 30
            else: x, y = -30, self.np_random.uniform(0, self.screen_height)
            
            pattern = self.np_random.choice(['roamer', 'shooter'])
            
            self.enemies.append({
                "pos": pygame.math.Vector2(x, y),
                "size": 10,
                "health": 1,
                "pattern": pattern,
                "target_pos": self._get_random_arena_pos(),
                "fire_cooldown": self.np_random.integers(60, 120),
            })

    def _get_random_arena_pos(self):
        return pygame.math.Vector2(
            self.np_random.uniform(self.ARENA_MARGIN, self.screen_width - self.ARENA_MARGIN),
            self.np_random.uniform(self.ARENA_MARGIN, self.screen_height - self.ARENA_MARGIN)
        )

    def _update_player(self, velocity):
        self.player["pos"] += velocity * self.player["speed"]
        self.player["pos"].x = np.clip(self.player["pos"].x, self.player["size"], self.screen_width - self.player["size"])
        self.player["pos"].y = np.clip(self.player["pos"].y, self.player["size"], self.screen_height - self.player["size"])
        
        if self.player["fire_cooldown"] > 0: self.player["fire_cooldown"] -= 1
        if self.player["hit_timer"] > 0: self.player["hit_timer"] -= 1
        if self.player["shield_timer"] > 0: self.player["shield_timer"] -= 1
        if self.player["fast_fire_timer"] > 0: self.player["fast_fire_timer"] -= 1

    def _update_enemies(self):
        base_speed = 1.0 + (self.enemies_defeated // 5) * 0.25

        for enemy in self.enemies:
            direction_to_target = enemy["target_pos"] - enemy["pos"]
            if direction_to_target.length() < 20:
                enemy["target_pos"] = self._get_random_arena_pos()
            
            enemy["pos"] += direction_to_target.normalize() * base_speed
            
            enemy["fire_cooldown"] -= 1
            if enemy["fire_cooldown"] <= 0 and enemy["pattern"] == 'shooter':
                # sfx: enemy_shoot.wav
                direction_to_player = (self.player["pos"] - enemy["pos"]).normalize()
                self._fire_projectile(enemy["pos"], direction_to_player, "enemy")
                enemy["fire_cooldown"] = self.np_random.integers(90, 150)

    def _update_projectiles(self):
        for p in self.projectiles:
            p["pos"] += p["vel"]
            p["trail"].append(p["pos"].copy())
            if len(p["trail"]) > 5:
                p["trail"].pop(0)
        
        self.projectiles = [p for p in self.projectiles if 0 < p["pos"].x < self.screen_width and 0 < p["pos"].y < self.screen_height]

    def _update_powerups(self):
        spawn_chance = 0.002 + (self.enemies_defeated // 10) * 0.001
        if self.np_random.random() < spawn_chance and len(self.powerups) < 2:
            self.powerups.append({
                "pos": self._get_random_arena_pos(),
                "type": self.np_random.choice(["shield", "fast_fire"]),
                "spawn_time": self.steps
            })
        
        self.powerups = [p for p in self.powerups if self.steps - p["spawn_time"] < 300] # 10 second lifetime

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _fire_projectile(self, pos, direction, owner):
        self.projectiles.append({
            "pos": pos.copy(),
            "vel": direction.normalize() * 10,
            "owner": owner,
            "trail": [],
        })
        # Muzzle flash
        flash_pos = pos + direction * (self.player["size"] if owner == "player" else self.enemies[0]["size"])
        self._create_explosion(flash_pos, 3, 0.2, 3)

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Enemies
        for proj in self.projectiles[:]:
            if proj["owner"] == "player":
                for enemy in self.enemies[:]:
                    if proj["pos"].distance_to(enemy["pos"]) < enemy["size"] + 4:
                        # sfx: enemy_hit.wav
                        reward += 0.1 # Hit reward
                        enemy["health"] -= 1
                        if enemy["health"] <= 0:
                            # sfx: enemy_explosion.wav
                            reward += 1.0 # Kill reward
                            self._create_explosion(enemy["pos"], 20, 1.0)
                            self.enemies.remove(enemy)
                            self.enemies_defeated += 1
                        if proj in self.projectiles: self.projectiles.remove(proj)
                        break
        
        # Enemy projectiles vs Player
        for proj in self.projectiles[:]:
            if proj["owner"] == "enemy":
                if proj["pos"].distance_to(self.player["pos"]) < self.player["size"] + 4:
                    if self.player["shield_timer"] <= 0:
                        # sfx: player_hit.wav
                        self.player["health"] -= 10
                        self.player["hit_timer"] = 10 # Flash for 10 frames
                    if proj in self.projectiles: self.projectiles.remove(proj)
        
        return reward
    
    def _handle_powerup_collection(self):
        reward = 0
        for powerup in self.powerups[:]:
            if self.player["pos"].distance_to(powerup["pos"]) < self.player["size"] + 10:
                # sfx: powerup_collect.wav
                reward += 0.5
                if powerup["type"] == "shield":
                    self.player["shield_timer"] = 300 # 10 seconds
                elif powerup["type"] == "fast_fire":
                    self.player["fast_fire_timer"] = 300 # 10 seconds
                self.powerups.remove(powerup)
        return reward

    def _create_explosion(self, pos, num_particles, speed_mult, lifetime=20):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(lifetime // 2, lifetime),
                "color": random.choice(self.COLOR_EXPLOSION)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Grid
        for x in range(0, self.screen_width, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.screen_width, y))

        # Powerups
        for p in self.powerups:
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            radius = int(8 + pulse * 4)
            color = self.COLOR_POWERUP_SHIELD if p["type"] == "shield" else self.COLOR_POWERUP_FIRERATE
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), radius, (*color, 100))
            pygame.gfxdraw.aacircle(self.screen, int(p["pos"].x), int(p["pos"].y), radius, color)

        # Enemies
        for enemy in self.enemies:
            pos_int = (int(enemy["pos"].x), int(enemy["pos"].y))
            size = enemy["size"]
            pulse = (math.sin(self.steps * 0.1) + 1) / 2
            glow_size = int(size * (1.5 + pulse * 0.5))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], glow_size, (*self.COLOR_ENEMY_GLOW, 100))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos_int[0] - size, pos_int[1] - size, size*2, size*2))

        # Player
        if self.player["health"] > 0:
            pos_int = (int(self.player["pos"].x), int(self.player["pos"].y))
            size = self.player["size"]
            
            if self.player["hit_timer"] > 0 and self.steps % 2 == 0:
                render_color = self.COLOR_WHITE
            else:
                render_color = self.COLOR_PLAYER
            
            # Shield effect
            if self.player["shield_timer"] > 0:
                alpha = int(50 + (self.player["shield_timer"] / 300) * 100)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], size + 8, (*self.COLOR_POWERUP_SHIELD, alpha))
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], size + 8, self.COLOR_POWERUP_SHIELD)

            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], size * 2, (*self.COLOR_PLAYER_GLOW, 150))

            # Body
            p1 = self.player["pos"] + self.player["aim_vector"] * size
            p2 = self.player["pos"] + self.player["aim_vector"].rotate(150) * size * 0.8
            p3 = self.player["pos"] + self.player["aim_vector"].rotate(-150) * size * 0.8
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], render_color)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], render_color)

        # Projectiles
        for p in self.projectiles:
            color = self.COLOR_PLAYER_BULLET if p["owner"] == "player" else self.COLOR_ENEMY_BULLET
            # Trail
            for i, trail_pos in enumerate(p["trail"]):
                alpha = int((i / len(p["trail"])) * 100)
                pygame.gfxdraw.filled_circle(self.screen, int(trail_pos.x), int(trail_pos.y), 2, (*color, alpha))
            # Head
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), 3, color)

        # Particles
        for p in self.particles:
            alpha = int((p["life"] / 20) * 255)
            alpha = max(0, min(255, alpha))
            size = int(p["life"] / 5)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), size, (*p["color"], alpha))

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.player["health"] / self.player["max_health"])
        bar_width = 200
        bar_height = 15
        pygame.draw.rect(self.screen, self.COLOR_ENEMY, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, 10, int(bar_width * health_pct), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_WHITE, (10, 10, bar_width, bar_height), 1)

        # Score
        score_text = self.game_font.render(f"SCORE: {int(self.score)}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (self.screen_width - score_text.get_width() - 10, 10))
        
        # Enemies Remaining
        enemies_text = self.game_font_small.render(f"ENEMIES: {len(self.enemies)}/{self.TOTAL_ENEMIES}", True, self.COLOR_ENEMY)
        self.screen.blit(enemies_text, (self.screen_width - enemies_text.get_width() - 10, 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "enemies_remaining": len(self.enemies),
        }
        
    def close(self):
        pygame.quit()

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set the video driver to a dummy one for headless execution
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Run a few random steps
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode finished.")
            obs, info = env.reset()

    env.close()
    print("Environment ran successfully.")

    # To visualize the game, you would need a different setup
    # that creates a real pygame window.
    # Example (requires a display):
    #
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    #
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # pygame.display.set_caption("Robot Arena")
    # screen = pygame.display.set_mode((640, 400))
    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #     
    #     # Simple keyboard controls for human play
    #     keys = pygame.key.get_pressed()
    #     mov = 0
    #     if keys[pygame.K_UP]: mov = 1
    #     elif keys[pygame.K_DOWN]: mov = 2
    #     elif keys[pygame.K_LEFT]: mov = 3
    #     elif keys[pygame.K_RIGHT]: mov = 4
    #     space = 1 if keys[pygame.K_SPACE] else 0
    #     shift = 1 if keys[pygame.K_LSHIFT] else 0
    #
    #     action = [mov, space, shift]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #
    #     # Draw the observation to the screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #
    #     if terminated or truncated:
    #         print(f"Game Over! Final Score: {info['score']}")
    #         obs, info = env.reset()
    #         pygame.time.wait(2000) # Pause before restarting
    #
    #     env.clock.tick(30) # Limit to 30 FPS
    #
    # env.close()