
# Generated: 2025-08-27T23:44:09.202687
# Source Brief: brief_03556.md
# Brief Index: 3556

        
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
        "Controls: Use ↑ and ↓ to move your ship. Press Space to fire your weapon. Survive the waves!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro side-scrolling shooter. Survive for three 60-second stages against increasingly difficult waves of geometric enemies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 50)
    COLOR_PLAYER_PROJECTILE = (255, 255, 255)
    COLOR_ENEMY_PROJECTILE = (255, 100, 100)
    ENEMY_COLORS = {
        1: (255, 80, 80),   # Red
        2: (255, 165, 0),  # Orange
        3: (255, 255, 0),  # Yellow
    }
    PARTICLE_COLORS = [(255, 255, 224), (255, 165, 0), (255, 69, 0)]
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEART = (255, 50, 50)

    # Player settings
    PLAYER_SPEED = 8
    PLAYER_FIRE_COOLDOWN = 6  # frames

    # Game settings
    TOTAL_STAGES = 3
    STAGE_DURATION_SECONDS = 60
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        self.stars = []
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._initialize_stars()

        # Player state
        self.player_pos = [50, self.HEIGHT // 2]
        self.player_lives = 3
        self.player_fire_cooldown_timer = 0
        self.prev_space_held = False

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.total_game_timer = self.TOTAL_STAGES * self.STAGE_DURATION_SECONDS * self.FPS
        self.stage_timer = self.STAGE_DURATION_SECONDS * self.FPS
        self.stage_clear_pause = 0

        # Dynamic objects
        self.player_projectiles = []
        self.enemies = []
        self.enemy_projectiles = []
        self.particles = []

        # Difficulty scaling
        self.enemy_spawn_timer = 0
        self._update_difficulty()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        self.steps += 1
        
        # --- Handle Pauses ---
        if self.stage_clear_pause > 0:
            self.stage_clear_pause -= 1
            if self.stage_clear_pause == 0:
                self.stage += 1
                if self.stage > self.TOTAL_STAGES:
                    self.game_over = True # Win condition
                else:
                    self._update_difficulty()
                    self.stage_timer = self.STAGE_DURATION_SECONDS * self.FPS
            
            terminated = self.game_over
            if terminated and self.stage > self.TOTAL_STAGES:
                reward += 300 # Win game bonus
            
            return self._get_observation(), reward, terminated, False, self._get_info()

        # --- Normal Step Logic ---
        reward += 0.01 # Small reward for surviving a step

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Handle player actions
        self._handle_player_input(movement, space_held)

        # Update game logic
        self._update_player_projectiles()
        self._update_enemies()
        self._update_enemy_projectiles()
        self._update_particles()
        
        # Handle collisions and collect rewards
        reward += self._handle_collisions()

        # Update timers and difficulty
        self.total_game_timer -= 1
        self.stage_timer -= 1
        
        # Difficulty ramp-up within a stage (every 20s)
        if (self.STAGE_DURATION_SECONDS * self.FPS - self.stage_timer) % (20 * self.FPS) == 0 and self.stage_timer < self.STAGE_DURATION_SECONDS * self.FPS:
             self.base_spawn_rate = max(15, self.base_spawn_rate * 0.95)
             self.base_speed += 0.2

        # Check for stage clear
        if self.stage_timer <= 0 and not self.game_over:
            reward += 100  # Stage clear bonus
            self.stage_clear_pause = self.FPS * 2 # 2 second pause
            self.enemies.clear()
            self.enemy_projectiles.clear()
            self.player_projectiles.clear()


        # Check termination conditions
        terminated = self.game_over or self.total_game_timer <= 0
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # Movement
        if movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
        self.player_pos[1] = np.clip(self.player_pos[1], 15, self.HEIGHT - 15)

        # Firing
        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1
        
        if space_held and not self.prev_space_held and self.player_fire_cooldown_timer == 0:
            # Fire projectile
            self.player_projectiles.append({
                "rect": pygame.Rect(self.player_pos[0] + 15, self.player_pos[1] - 2, 12, 4),
                "speed": 15
            })
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN
            # sfx: player_shoot.wav
        
        self.prev_space_held = space_held

    def _update_player_projectiles(self):
        for p in self.player_projectiles:
            p["rect"].x += p["speed"]
        self.player_projectiles = [p for p in self.player_projectiles if p["rect"].left < self.WIDTH]

    def _update_enemies(self):
        # Spawn new enemies
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            self._spawn_enemy()
            self.enemy_spawn_timer = self.base_spawn_rate

        # Move existing enemies
        for e in self.enemies:
            e["pos"][0] -= e["speed"]
            if e["type"] == 2: # Sine wave
                e["pos"][1] = e["start_y"] + math.sin(e["pos"][0] / 50) * 40
            elif e["type"] == 3: # Homing
                 e["pos"][1] += np.clip(self.player_pos[1] - e["pos"][1], -e["speed"]*0.5, e["speed"]*0.5)

            e["rect"].center = e["pos"]

            # Enemy firing
            if "fire_rate" in e:
                e["fire_timer"] -= 1
                if e["fire_timer"] <= 0:
                    self.enemy_projectiles.append({
                        "rect": pygame.Rect(e["rect"].centerx, e["rect"].centery, 10, 10),
                        "speed": -6,
                    })
                    e["fire_timer"] = e["fire_rate"] + self.np_random.integers(-30, 30)
                    # sfx: enemy_shoot.wav

        self.enemies = [e for e in self.enemies if e["rect"].right > 0]

    def _spawn_enemy(self):
        enemy_type = self.np_random.choice(range(1, min(self.stage, 3) + 1))
        start_y = self.np_random.integers(20, self.HEIGHT - 20)
        
        enemy = {
            "pos": [self.WIDTH + 20, start_y],
            "start_y": start_y,
            "speed": self.base_speed + self.np_random.random() * 1.5,
            "type": enemy_type,
            "size": 15 + (enemy_type * 2),
            "color": self.ENEMY_COLORS[enemy_type],
        }

        if enemy_type == 1:
            enemy["rect"] = pygame.Rect(0, 0, enemy["size"], enemy["size"])
        elif enemy_type == 2:
            enemy["rect"] = pygame.Rect(0, 0, enemy["size"], enemy["size"])
        elif enemy_type == 3:
            enemy["rect"] = pygame.Rect(0, 0, enemy["size"], enemy["size"])
            enemy["fire_rate"] = 120 - (self.stage * 10)
            enemy["fire_timer"] = self.np_random.integers(0, enemy["fire_rate"])

        enemy["rect"].center = enemy["pos"]
        self.enemies.append(enemy)

    def _update_enemy_projectiles(self):
        for p in self.enemy_projectiles:
            p["rect"].x += p["speed"]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if p["rect"].right > 0]

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["radius"] -= 0.2
        self.particles = [p for p in self.particles if p["life"] > 0 and p["radius"] > 0]

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 10, 20, 20)

        # Player projectiles vs enemies
        for p in self.player_projectiles[:]:
            for e in self.enemies[:]:
                if p["rect"].colliderect(e["rect"]):
                    self._create_explosion(e["rect"].center, e["color"])
                    self.player_projectiles.remove(p)
                    self.enemies.remove(e)
                    self.score += 1
                    reward += 1
                    # sfx: explosion.wav
                    break

        # Enemy projectiles vs player
        for p in self.enemy_projectiles[:]:
            if player_rect.colliderect(p["rect"]):
                self.enemy_projectiles.remove(p)
                reward += self._handle_player_hit()
                break

        # Enemies vs player
        for e in self.enemies[:]:
            if player_rect.colliderect(e["rect"]):
                self._create_explosion(e["rect"].center, e["color"])
                self.enemies.remove(e)
                reward += self._handle_player_hit()
                break
        
        return reward

    def _handle_player_hit(self):
        self.player_lives -= 1
        self._create_explosion(self.player_pos, self.COLOR_PLAYER, 40)
        # sfx: player_hit.wav
        if self.player_lives <= 0:
            self.game_over = True
        return -10

    def _create_explosion(self, pos, base_color, num_particles=20):
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 4
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "radius": self.np_random.random() * 4 + 2,
                "color": random.choice(self.PARTICLE_COLORS)
            })

    def _update_difficulty(self):
        self.base_spawn_rate = 100 - (self.stage * 15)
        self.base_speed = 2.5 + (self.stage * 0.75)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for star in self.stars:
            star[0] -= star[2]
            if star[0] < 0:
                star[0] = self.WIDTH
                star[1] = self.np_random.integers(0, self.HEIGHT)
            pygame.draw.circle(self.screen, star[3], (int(star[0]), int(star[1])), int(star[4]))

        # Particles
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), p["color"])

        # Player projectiles
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJECTILE, p["rect"])
            pygame.draw.line(self.screen, self.COLOR_PLAYER_GLOW, p["rect"].midleft, p["rect"].midright, 6)

        # Enemy projectiles
        for p in self.enemy_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, p["rect"].centerx, p["rect"].centery, p["rect"].width // 2, self.COLOR_ENEMY_PROJECTILE)

        # Enemies
        for e in self.enemies:
            pos = (int(e["rect"].centerx), int(e["rect"].centery))
            size = e["size"]
            if e["type"] == 1: # Square
                pygame.gfxdraw.box(self.screen, e["rect"], e["color"])
            elif e["type"] == 2: # Diamond
                points = [(pos[0], pos[1] - size//2), (pos[0] + size//2, pos[1]), (pos[0], pos[1] + size//2), (pos[0] - size//2, pos[1])]
                pygame.gfxdraw.aapolygon(self.screen, points, e["color"])
                pygame.gfxdraw.filled_polygon(self.screen, points, e["color"])
            elif e["type"] == 3: # Pentagon
                points = []
                for i in range(5):
                    angle = (math.pi * 2 / 5) * i - math.pi / 2
                    points.append((pos[0] + math.cos(angle) * size//2, pos[1] + math.sin(angle) * size//2))
                pygame.gfxdraw.aapolygon(self.screen, points, e["color"])
                pygame.gfxdraw.filled_polygon(self.screen, points, e["color"])

        # Player
        if self.player_lives > 0:
            p = self.player_pos
            points = [(p[0] + 15, p[1]), (p[0] - 10, p[1] - 10), (p[0] - 10, p[1] + 10)]
            # Glow
            glow_points = [(p[0] + 20, p[1]), (p[0] - 15, p[1] - 15), (p[0] - 15, p[1] + 15)]
            pygame.gfxdraw.filled_trigon(self.screen, int(glow_points[0][0]), int(glow_points[0][1]), int(glow_points[1][0]), int(glow_points[1][1]), int(glow_points[2][0]), int(glow_points[2][1]), self.COLOR_PLAYER_GLOW)
            # Ship
            pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_PLAYER)
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_PLAYER)

    def _render_ui(self):
        # Lives
        for i in range(self.player_lives):
            self._draw_heart(25 + i * 35, 25, 12)

        # Timer
        secs = self.stage_timer // self.FPS
        timer_text = f"{secs:02d}"
        text_surf = self.font_small.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - 40, 15))

        # Score
        score_text = f"SCORE: {self.score}"
        text_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH // 2 - text_surf.get_width() // 2, 15))

        # Stage
        stage_text = f"STAGE {self.stage}"
        text_surf = self.font_small.render(stage_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH // 2 - text_surf.get_width() // 2, self.HEIGHT - 30))
        
        # Stage Clear Message
        if self.stage_clear_pause > 0 and self.stage <= self.TOTAL_STAGES:
            msg = "STAGE CLEAR" if self.stage < self.TOTAL_STAGES else "VICTORY!"
            text_surf = self.font_large.render(msg, True, self.COLOR_PLAYER)
            self.screen.blit(text_surf, (self.WIDTH // 2 - text_surf.get_width() // 2, self.HEIGHT // 2 - text_surf.get_height() // 2))

        # Game Over Message
        if self.game_over and self.player_lives <= 0:
            text_surf = self.font_large.render("GAME OVER", True, self.COLOR_HEART)
            self.screen.blit(text_surf, (self.WIDTH // 2 - text_surf.get_width() // 2, self.HEIGHT // 2 - text_surf.get_height() // 2))

    def _draw_heart(self, x, y, size):
        pygame.gfxdraw.filled_circle(self.screen, x + size // 4, y - size // 4, size // 3, self.COLOR_HEART)
        pygame.gfxdraw.aacircle(self.screen, x + size // 4, y - size // 4, size // 3, self.COLOR_HEART)
        pygame.gfxdraw.filled_circle(self.screen, x + 3 * size // 4, y - size // 4, size // 3, self.COLOR_HEART)
        pygame.gfxdraw.aacircle(self.screen, x + 3 * size // 4, y - size // 4, size // 3, self.COLOR_HEART)
        points = [(x, y), (x + size, y), (x + size // 2, y + size // 2 + size//4)]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HEART)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "stage": self.stage,
        }

    def _initialize_stars(self):
        self.stars = []
        for _ in range(150):
            speed = 0.5 + self.np_random.random() * 1.5
            size = 1 if speed < 1.2 else 2
            color_val = 50 + int(speed * 40)
            self.stars.append([
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                speed,
                (color_val, color_val, color_val + 10),
                size
            ])
        self.stars.sort(key=lambda s: s[2]) # Draw slow stars first

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

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # move: 0=none, 1=up, 2=down
    # space: 0=released, 1=held
    # shift: 0=released, 1=held
    action = [0, 0, 0]

    # Pygame window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    print(env.user_guide)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            # Key Down
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
            
            # Key Up
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP and action[0] == 1:
                    action[0] = 0
                elif event.key == pygame.K_DOWN and action[0] == 2:
                    action[0] = 0
                elif event.key == pygame.K_SPACE:
                    action[1] = 0
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    
    # Keep window open for a few seconds to show final message
    end_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - end_time < 3000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        else:
            continue
        break
        
    env.close()