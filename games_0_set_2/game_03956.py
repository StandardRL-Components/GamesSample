
# Generated: 2025-08-28T00:57:32.520111
# Source Brief: brief_03956.md
# Brief Index: 3956

        
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
        "Controls: ←→ to move. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of descending aliens in a retro side-scrolling shooter for 3 minutes."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.WIN_TIME_SECONDS = 180

        # Colors
        self.COLOR_BG = (15, 15, 35)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_PROJ = (100, 150, 255)
        self.COLOR_ALIEN_1 = (255, 50, 50)
        self.COLOR_ALIEN_2 = (255, 150, 50)
        self.COLOR_ALIEN_PROJ = (255, 50, 255)
        self.COLOR_EXPLOSION = [(255, 255, 50), (255, 150, 50), (255, 255, 255)]
        self.COLOR_SCORE = (220, 220, 220)
        self.COLOR_TIMER = (220, 220, 220)
        
        # Player settings
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN = 6 # frames

        # Projectile settings
        self.PROJ_SPEED_PLAYER = 12
        self.PROJ_SPEED_ALIEN = 5

        # Alien settings
        self.INITIAL_ALIEN_SPAWN_INTERVAL = 60 # frames
        self.INITIAL_ALIEN_SPEED = 1.0
        self.INITIAL_ALIEN_FIRE_PROB = 0.01

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_pos = [0, 0]
        self.player_fire_cooldown_timer = 0
        self.player_projectiles = []
        self.aliens = []
        self.alien_projectiles = []
        self.particles = []
        self.stars = []
        self.game_timer = 0
        self.spawn_timer = 0
        self.alien_spawn_interval = 0
        self.alien_speed = 0
        self.alien_fire_prob = 0
        self.difficulty_tier = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 40]
        self.player_fire_cooldown_timer = 0
        
        self.player_projectiles = []
        self.aliens = []
        self.alien_projectiles = []
        self.particles = []
        
        self.game_timer = self.WIN_TIME_SECONDS
        self.spawn_timer = self.INITIAL_ALIEN_SPAWN_INTERVAL
        
        # Difficulty settings
        self.alien_spawn_interval = self.INITIAL_ALIEN_SPAWN_INTERVAL
        self.alien_speed = self.INITIAL_ALIEN_SPEED
        self.alien_fire_prob = self.INITIAL_ALIEN_FIRE_PROB
        self.difficulty_tier = 0

        if not self.stars:
            self._create_stars()
        
        return self._get_observation(), self._get_info()

    def _create_stars(self):
        self.stars = []
        for _ in range(150):
            speed = self.np_random.uniform(0.2, 1.5)
            brightness = int(50 + speed * 100)
            self.stars.append({
                "pos": [self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)],
                "speed": speed,
                "color": (brightness, brightness, brightness + 20)
            })

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0.0
        terminated = False

        if not self.game_over:
            self.steps += 1
            reward += 0.01  # Small reward for surviving each frame

            self._update_difficulty_and_timer()
            self._handle_input(action)
            
            self._update_player_projectiles()
            self._update_aliens()
            self._update_alien_projectiles()
            
            collision_reward = self._handle_collisions()
            reward += collision_reward

            self._update_particles()
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over: # Max steps reached without dying
            self.game_over = True
        
        if self.game_over:
            if self.win:
                reward = 100.0
            else:
                reward = -100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_difficulty_and_timer(self):
        self.game_timer -= 1 / self.FPS
        if self.game_timer <= 0:
            self.game_timer = 0
            self.win = True
            self.game_over = True
            return

        # Increase difficulty every 30 seconds
        new_tier = (self.WIN_TIME_SECONDS - int(self.game_timer)) // 30
        if new_tier > self.difficulty_tier:
            self.difficulty_tier = new_tier
            self.alien_speed += 0.5
            self.alien_fire_prob = min(self.alien_fire_prob + 0.005, 0.1)
            self.alien_spawn_interval = max(self.alien_spawn_interval - 5, 15)
            # print(f"Difficulty Up! Speed: {self.alien_speed}, Fire Prob: {self.alien_fire_prob}, Spawn: {self.alien_spawn_interval}") # For debugging

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        # Player movement
        if movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)
        
        # Player firing
        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1
        
        if space_held and self.player_fire_cooldown_timer == 0:
            # SFX: Player shoot
            self.player_projectiles.append(list(self.player_pos))
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_player_projectiles(self):
        for proj in self.player_projectiles:
            proj[1] -= self.PROJ_SPEED_PLAYER
        self.player_projectiles = [p for p in self.player_projectiles if p[1] > 0]

    def _update_aliens(self):
        # Spawn new aliens
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self.spawn_timer = self.alien_spawn_interval
            pos_x = self.np_random.integers(40, self.WIDTH - 40)
            alien_type = self.np_random.choice([1, 2])
            self.aliens.append({
                "pos": [pos_x, -20],
                "type": alien_type,
                "bob_offset": self.np_random.uniform(0, 2 * math.pi)
            })
        
        # Update existing aliens
        for alien in self.aliens:
            alien["pos"][1] += self.alien_speed
            # Add a slight side-to-side bobbing motion
            alien["pos"][0] += math.sin(self.steps * 0.1 + alien["bob_offset"]) * 0.5
            
            # Alien firing
            if self.np_random.random() < self.alien_fire_prob:
                # SFX: Alien shoot
                self.alien_projectiles.append(list(alien["pos"]))
        
        self.aliens = [a for a in self.aliens if a["pos"][1] < self.HEIGHT + 20]

    def _update_alien_projectiles(self):
        for proj in self.alien_projectiles:
            proj[1] += self.PROJ_SPEED_ALIEN
        self.alien_projectiles = [p for p in self.alien_projectiles if p[1] < self.HEIGHT]

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                dist = math.hypot(proj[0] - alien["pos"][0], proj[1] - alien["pos"][1])
                if dist < 20:
                    # SFX: Explosion
                    self._create_explosion(alien["pos"])
                    self.player_projectiles.remove(proj)
                    self.aliens.remove(alien)
                    self.score += 1
                    reward += 1.0
                    break
        
        # Alien projectiles vs Player
        player_rect = pygame.Rect(self.player_pos[0] - 15, self.player_pos[1] - 10, 30, 20)
        for proj in self.alien_projectiles[:]:
            if player_rect.collidepoint(proj[0], proj[1]):
                # SFX: Player death explosion
                self._create_explosion(self.player_pos, num_particles=100, scale=2.0)
                self.alien_projectiles.remove(proj)
                self.game_over = True
                break
        
        return reward

    def _create_explosion(self, pos, num_particles=30, scale=1.0):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1 * scale, 5 * scale)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            color = random.choice(self.COLOR_EXPLOSION)
            self.particles.append({
                "pos": list(pos), 
                "vel": velocity, 
                "life": lifetime,
                "max_life": lifetime,
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95 # friction
            p["vel"][1] *= 0.95
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]
    
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
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": round(self.game_timer, 2),
            "win": self.win
        }

    def _render_game(self):
        # Render stars
        for star in self.stars:
            star["pos"][1] += star["speed"]
            if star["pos"][1] > self.HEIGHT:
                star["pos"][0] = self.np_random.integers(0, self.WIDTH)
                star["pos"][1] = 0
            pygame.draw.circle(self.screen, star["color"], (int(star["pos"][0]), int(star["pos"][1])), 1)

        # Render player projectiles
        for proj in self.player_projectiles:
            pygame.draw.line(self.screen, self.COLOR_PLAYER_PROJ, (int(proj[0]), int(proj[1])), (int(proj[0]), int(proj[1]) - 10), 3)

        # Render alien projectiles
        for proj in self.alien_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj[0]), int(proj[1]), 4, self.COLOR_ALIEN_PROJ)
            pygame.gfxdraw.aacircle(self.screen, int(proj[0]), int(proj[1]), 4, self.COLOR_ALIEN_PROJ)

        # Render aliens
        for alien in self.aliens:
            x, y = int(alien["pos"][0]), int(alien["pos"][1])
            color = self.COLOR_ALIEN_1 if alien["type"] == 1 else self.COLOR_ALIEN_2
            points = [(x, y-10), (x-15, y+10), (x+15, y+10)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Render player
        if not self.game_over or self.win:
            x, y = int(self.player_pos[0]), int(self.player_pos[1])
            ship_points = [(x, y - 15), (x - 15, y + 10), (x + 15, y + 10)]
            pygame.gfxdraw.aapolygon(self.screen, ship_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, ship_points, self.COLOR_PLAYER)
            # Engine flare
            flare_y = y + 10 + self.np_random.integers(2, 8)
            flare_pts = [(x-7, y+10), (x+7, y+10), (x, flare_y)]
            pygame.gfxdraw.aapolygon(self.screen, flare_pts, (255, 200, 0))
            pygame.gfxdraw.filled_polygon(self.screen, flare_pts, (255, 200, 0))

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (p["color"][0], p["color"][1], p["color"][2])
            size = int(3 * (p["life"] / p["max_life"]))
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p["pos"][0]), int(p["pos"][1])), size)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 10))

        # Timer
        mins, secs = divmod(int(self.game_timer), 60)
        timer_text = self.font_ui.render(f"TIME: {mins:02}:{secs:02}", True, self.COLOR_TIMER)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over / Win Message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = (50, 255, 50)
            else:
                msg = "GAME OVER"
                color = (255, 50, 50)
            
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

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

# Example usage:
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # To run the game with keyboard controls
    import pygame
    pygame.display.set_caption("Galactic Survivor")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        move_action = 0 # No-op
        if keys[pygame.K_LEFT]:
            move_action = 3
        elif keys[pygame.K_RIGHT]:
            move_action = 4
            
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        # --- End Human Controls ---

        # For random agent: action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()