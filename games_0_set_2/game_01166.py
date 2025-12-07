import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. Hold Space to jump. Hold Shift to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a jumping robot in a side-scrolling action environment to defeat waves of enemies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (15, 15, 35)
    COLOR_GROUND = (40, 30, 30)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_EYE = (255, 255, 255)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_PLAYER_PROJ = (255, 255, 100)
    COLOR_ENEMY_PROJ = (255, 150, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    
    # Screen and World
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_WIDTH = 1280
    GROUND_Y = 350
    
    # Player Physics
    PLAYER_SPEED = 5
    JUMP_POWER = 12
    GRAVITY = 0.6
    PLAYER_WIDTH, PLAYER_HEIGHT = 20, 30
    PLAYER_SHOOT_COOLDOWN = 8 # frames
    
    # Enemy
    ENEMY_WIDTH, ENEMY_HEIGHT = 24, 24
    ENEMY_SPEED = 1.2
    
    # Game Rules
    MAX_STEPS = 10000
    MAX_WAVES = 10
    INITIAL_PLAYER_HEALTH = 3
    ENEMIES_PER_WAVE = 7

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # RNG
        self.np_random = None

        # Initialize state variables
        self.player = {}
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.camera_x = 0
        
        # self.reset() is called by the gym wrapper, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None: # Initialize RNG if it's the first time
            self.np_random = np.random.default_rng(seed=seed)

        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False

        self.player = {
            "pos": pygame.Vector2(self.WORLD_WIDTH / 2, self.GROUND_Y - self.PLAYER_HEIGHT),
            "vel": pygame.Vector2(0, 0),
            "width": self.PLAYER_WIDTH,
            "height": self.PLAYER_HEIGHT,
            "health": self.INITIAL_PLAYER_HEALTH,
            "on_ground": True,
            "facing_dir": 1, # 1 for right, -1 for left
            "shoot_cooldown": 0,
        }
        
        self.player_projectiles.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty for each frame to encourage speed
        
        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._update_enemies()
            self._update_projectiles()
            self._update_particles()
            reward += self._handle_collisions()
            reward += self._check_wave_clear()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            if self.player["health"] <= 0:
                reward = -10 # Game over penalty
            self.game_over = True
        
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
        
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal Movement
        if movement == 3: # Left
            self.player["vel"].x = -self.PLAYER_SPEED
            self.player["facing_dir"] = -1
        elif movement == 4: # Right
            self.player["vel"].x = self.PLAYER_SPEED
            self.player["facing_dir"] = 1
        else:
            self.player["vel"].x = 0
            
        # Jump
        if space_held and self.player["on_ground"]:
            self.player["vel"].y = -self.JUMP_POWER
            self.player["on_ground"] = False
            # Sound: jump
            self._create_particles(self.player["pos"] + pygame.Vector2(self.PLAYER_WIDTH/2, self.PLAYER_HEIGHT), 5, self.COLOR_GROUND, 2, 8)

        # Shoot
        if self.player["shoot_cooldown"] > 0:
            self.player["shoot_cooldown"] -= 1
        
        if shift_held and self.player["shoot_cooldown"] <= 0:
            self.player["shoot_cooldown"] = self.PLAYER_SHOOT_COOLDOWN
            proj_start_pos = self.player["pos"] + pygame.Vector2(self.PLAYER_WIDTH / 2, self.PLAYER_HEIGHT / 2)
            proj_vel = pygame.Vector2(self.player["facing_dir"] * 15, 0)
            self.player_projectiles.append({"pos": proj_start_pos, "vel": proj_vel})
            # Sound: player shoot
            self._create_particles(proj_start_pos, 3, self.COLOR_PLAYER_PROJ, 3, 5)

    def _update_player(self):
        # Apply gravity
        self.player["vel"].y += self.GRAVITY
        
        # Update position
        self.player["pos"] += self.player["vel"]
        
        # Ground collision
        if self.player["pos"].y + self.PLAYER_HEIGHT > self.GROUND_Y:
            self.player["pos"].y = self.GROUND_Y - self.PLAYER_HEIGHT
            self.player["vel"].y = 0
            if not self.player["on_ground"]:
                # Sound: land
                self.player["on_ground"] = True

        # World bounds
        self.player["pos"].x = max(0, min(self.player["pos"].x, self.WORLD_WIDTH - self.PLAYER_WIDTH))
    
    def _spawn_wave(self):
        self.enemies.clear()
        for i in range(self.ENEMIES_PER_WAVE):
            side = self.np_random.choice([-1, 1])
            dist = self.np_random.integers(200, 500)
            x_pos = self.player["pos"].x + side * dist
            x_pos = max(0, min(x_pos, self.WORLD_WIDTH - self.ENEMY_WIDTH))

            shoot_interval = max(10, 25 - (self.wave - 1))
            
            self.enemies.append({
                "pos": pygame.Vector2(x_pos, self.GROUND_Y - self.ENEMY_HEIGHT),
                "width": self.ENEMY_WIDTH, "height": self.ENEMY_HEIGHT,
                "shoot_cooldown": self.np_random.integers(0, shoot_interval),
                "shoot_interval": shoot_interval,
            })

    def _update_enemies(self):
        for enemy in self.enemies:
            # Move towards player
            direction = 1 if self.player["pos"].x > enemy["pos"].x else -1
            enemy["pos"].x += direction * self.ENEMY_SPEED
            
            # Shoot
            enemy["shoot_cooldown"] -= 1
            if enemy["shoot_cooldown"] <= 0:
                enemy["shoot_cooldown"] = enemy["shoot_interval"]
                proj_start_pos = enemy["pos"] + pygame.Vector2(self.ENEMY_WIDTH / 2, self.ENEMY_HEIGHT)
                self.enemy_projectiles.append({"pos": proj_start_pos, "vel": pygame.Vector2(0, 8)})
                # Sound: enemy shoot
    
    def _update_projectiles(self):
        for proj_list in [self.player_projectiles, self.enemy_projectiles]:
            for proj in proj_list[:]:
                proj["pos"] += proj["vel"]
                if not (0 < proj["pos"].x < self.WORLD_WIDTH and 0 < proj["pos"].y < self.SCREEN_HEIGHT):
                    proj_list.remove(proj)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs enemies
        for p_proj in self.player_projectiles[:]:
            p_rect = pygame.Rect(p_proj["pos"].x, p_proj["pos"].y, 8, 4)
            for enemy in self.enemies[:]:
                e_rect = pygame.Rect(enemy["pos"].x, enemy["pos"].y, enemy["width"], enemy["height"])
                if p_rect.colliderect(e_rect):
                    reward += 0.1 # Hit bonus
                    reward += 1.0 # Kill bonus
                    self.score += 1
                    self.enemies.remove(enemy)
                    if p_proj in self.player_projectiles: self.player_projectiles.remove(p_proj)
                    # Sound: explosion
                    self._create_particles(enemy["pos"] + pygame.Vector2(enemy["width"]/2, enemy["height"]/2), 20, self.COLOR_ENEMY, 4, 20)
                    break
        
        # Enemy projectiles vs player
        player_rect = pygame.Rect(self.player["pos"], (self.player["width"], self.player["height"]))
        for e_proj in self.enemy_projectiles[:]:
            e_rect = pygame.Rect(e_proj["pos"].x, e_proj["pos"].y, 4, 8)
            if player_rect.colliderect(e_rect):
                self.player["health"] -= 1
                self.enemy_projectiles.remove(e_proj)
                # Sound: player hit
                self._create_particles(self.player["pos"] + pygame.Vector2(self.player["width"]/2, self.player["height"]/2), 15, self.COLOR_PLAYER, 3, 15)
                break
                
        return reward
        
    def _check_wave_clear(self):
        if not self.enemies:
            self.wave += 1
            if self.wave <= self.MAX_WAVES:
                self._spawn_wave()
                return 10.0 # Wave clear bonus
        return 0.0

    def _check_termination(self):
        return (
            self.player["health"] <= 0 or
            self.wave > self.MAX_WAVES
        )
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.player["health"],
        }
        
    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _create_particles(self, pos, count, color, speed_max, lifespan_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(lifespan_max // 2, lifespan_max)
            self.particles.append({"pos": pos.copy(), "vel": vel, "color": color, "lifespan": lifespan, "max_lifespan": lifespan})

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        
        # Update camera
        self.camera_x = self.player["pos"].x - self.SCREEN_WIDTH / 2
        self.camera_x = max(0, min(self.camera_x, self.WORLD_WIDTH - self.SCREEN_WIDTH))
        
        # Render background
        self._render_background()
        
        # Render ground
        ground_rect = pygame.Rect(0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = (*p["color"], alpha)
            size = 1 + int(2 * (p["lifespan"] / p["max_lifespan"]))
            pos_on_screen = (int(p["pos"].x - self.camera_x), int(p["pos"].y))
            try:
                pygame.gfxdraw.filled_circle(self.screen, pos_on_screen[0], pos_on_screen[1], size, color)
            except TypeError: # Sometimes color can have invalid alpha
                pass


        # Render projectiles
        for proj in self.player_projectiles:
            start = proj["pos"] - pygame.Vector2(self.camera_x, 0)
            end = start + proj["vel"].normalize() * 10
            pygame.draw.line(self.screen, self.COLOR_PLAYER_PROJ, start, end, 3)
        for proj in self.enemy_projectiles:
            rect = pygame.Rect(proj["pos"].x - self.camera_x, proj["pos"].y, 4, 8)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_PROJ, rect)
            
        # Render enemies
        for enemy in self.enemies:
            rect = pygame.Rect(enemy["pos"].x - self.camera_x, enemy["pos"].y, enemy["width"], enemy["height"])
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect)
            
        # Render player
        player_rect = pygame.Rect(self.player["pos"].x - self.camera_x, self.player["pos"].y, self.player["width"], self.player["height"])
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        eye_x = player_rect.centerx + self.player["facing_dir"] * 5
        eye_y = player_rect.centery - 5
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_EYE, (int(eye_x), int(eye_y)), 3)
        
        # Render UI
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

    def _render_background(self):
        # Far buildings (parallax factor 0.2)
        for i in range(0, self.WORLD_WIDTH, 100):
            if (i + 150) % 250 == 0:
                h = np.random.default_rng(seed=i).integers(50, 150)
                w = np.random.default_rng(seed=i+1).integers(40, 80)
                x = i - self.camera_x * 0.2
                color = (25, 25, 50)
                pygame.draw.rect(self.screen, color, (x, self.GROUND_Y - h, w, h))

        # Near buildings (parallax factor 0.5)
        for i in range(0, self.WORLD_WIDTH, 80):
             if (i + 50) % 150 == 0:
                h = np.random.default_rng(seed=i*2).integers(80, 200)
                w = np.random.default_rng(seed=i*2+1).integers(50, 90)
                x = i - self.camera_x * 0.5
                color = (35, 35, 60)
                pygame.draw.rect(self.screen, color, (x, self.GROUND_Y - h, w, h))

    def _render_ui(self):
        health_text = self.font_ui.render(f"HEALTH: {max(0, self.player['health'])}", True, self.COLOR_UI_TEXT)
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        wave_text = self.font_ui.render(f"WAVE: {self.wave}", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(health_text, (10, 10))
        self.screen.blit(score_text, (10, 35))
        self.screen.blit(wave_text, (10, 60))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        
        if self.wave > self.MAX_WAVES:
            text_str = "YOU WIN!"
        else:
            text_str = "GAME OVER"
        
        text_surf = self.font_game_over.render(text_str, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # The environment is designed to run headless, so this is the primary way to use it.
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Test a few random steps
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Reward={reward}, Terminated={terminated}, Truncated={truncated}, Info={info}")
        if terminated or truncated:
            print(f"Episode finished. Final info: {info}")
            obs, info = env.reset(seed=42+i+1)
            
    env.close()
    print("\nHeadless execution finished.")

    # Example for interactive play (requires a display)
    # To run this, you might need to comment out `os.environ["SDL_VIDEODRIVER"] = "dummy"`
    # in the __init__ method, or unset it before creating the GameEnv.
    # try:
    #     if "SDL_VIDEODRIVER" in os.environ:
    #         del os.environ["SDL_VIDEODRIVER"]
        
    #     pygame.display.init()
    #     pygame.font.init()

    #     env = GameEnv(render_mode="rgb_array")
    #     obs, info = env.reset()

    #     running = True
    #     screen = pygame.display.set_mode((640, 400))
    #     clock = pygame.time.Clock()
    #     print("\nStarting interactive mode. Close the window to exit.")

    #     while running:
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 running = False

    #         keys = pygame.key.get_pressed()
            
    #         movement = 0 # none
    #         if keys[pygame.K_LEFT]:
    #             movement = 3
    #         elif keys[pygame.K_RIGHT]:
    #             movement = 4
            
    #         space_held = 1 if keys[pygame.K_SPACE] else 0
    #         shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
    #         action = [movement, space_held, shift_held]

    #         obs, reward, terminated, truncated, info = env.step(action)

    #         if terminated or truncated:
    #             print(f"Game Over! Score: {info['score']}, Wave: {info['wave']}")
    #             # Display final frame
    #             surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #             screen.blit(surf, (0, 0))
    #             pygame.display.flip()
    #             pygame.time.wait(2000)
    #             obs, info = env.reset()
            
    #         # Display the observation from the environment
    #         surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #         screen.blit(surf, (0, 0))
    #         pygame.display.flip()
            
    #         clock.tick(30)
        
    #     env.close()
    # except pygame.error as e:
    #     print(f"\nCould not start interactive mode: {e}")
    #     print("This is expected if you are in a headless environment.")
    #     print("To run interactively, ensure you have a display configured.")