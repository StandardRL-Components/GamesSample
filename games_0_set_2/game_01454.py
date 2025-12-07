
# Generated: 2025-08-27T17:11:18.217057
# Source Brief: brief_01454.md
# Brief Index: 1454

        
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
        "Controls: Press space to jump and evade the asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Evade incoming asteroids by precisely timing your jumps in this retro side-scrolling space hopper."
    )

    # Frames auto-advance for this real-time game.
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_STAR = (200, 200, 220)
    COLOR_GROUND = (80, 80, 90)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (150, 255, 150, 50)
    COLOR_ASTEROID = [(120, 120, 130), (140, 140, 150), (100, 100, 110)]
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_PARTICLE = (200, 200, 200)

    # Player
    PLAYER_X = 100
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT = 40
    GRAVITY = 0.8
    JUMP_STRENGTH = -15

    # Game
    MAX_STEPS = 1800  # 60 seconds * 30 FPS
    STAGE_DURATION = 600 # 20 seconds * 30 FPS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        self.total_game_time_seconds = 0

        self.player_y = 0
        self.player_vy = 0
        self.player_rect = pygame.Rect(0,0,0,0)
        self.on_ground = True
        self.last_space_state = 0
        
        self.asteroids = []
        self.particles = []
        self.stars = []

        self.base_asteroid_speed = 0
        self.asteroid_spawn_timer_range = (0,0)
        self.asteroid_spawn_timer = 0
        
        self.reset()
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        self.total_game_time_seconds = self.MAX_STEPS / self.FPS
        
        # Reset player
        self.ground_y = self.SCREEN_HEIGHT - 40
        self.player_y = self.ground_y - self.PLAYER_HEIGHT
        self.player_vy = 0
        self.on_ground = True
        self.last_space_state = 0

        # Reset entities
        self.asteroids.clear()
        self.particles.clear()
        
        # Reset difficulty
        self._update_difficulty()

        # Generate static background
        self.stars = [
            (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT))
            for _ in range(150)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            self.steps += 1
            
            # --- Action Handling ---
            space_pressed = action[1] == 1
            
            # Jump on rising edge of space press
            if space_pressed and not self.last_space_state and self.on_ground:
                self.player_vy = self.JUMP_STRENGTH
                self.on_ground = False
                # Sound: Jump
                self._create_particles(self.player_rect.midbottom, 15, -2) # Jump dust
            self.last_space_state = space_pressed

            # --- Game Logic Update ---
            self._update_player()
            self._update_asteroids()
            self._update_particles()
            self._handle_spawning()
            
            # --- Reward & Termination ---
            reward += 0.01  # Small survival reward

            if self._check_collisions():
                self.game_over = True
                terminated = True
                reward = -10.0
                # Sound: Explosion / Player death
            else:
                # Stage progression
                if self.steps > 0 and self.steps % self.STAGE_DURATION == 0:
                    if self.steps >= self.MAX_STEPS: # Game won
                        self.game_over = True
                        terminated = True
                        reward += 100.0
                        # Sound: Victory
                    else: # Stage cleared
                        self.current_stage += 1
                        reward += 10.0
                        self._update_difficulty()
                        # Sound: Stage Clear
            
                # Check for max steps termination
                if self.steps >= self.MAX_STEPS:
                    self.game_over = True
                    terminated = True

            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_difficulty(self):
        """Sets difficulty parameters based on the current stage."""
        if self.current_stage == 1:
            self.base_asteroid_speed = 4.0
            self.asteroid_spawn_timer_range = (25, 50)
        elif self.current_stage == 2:
            self.base_asteroid_speed = 5.5
            self.asteroid_spawn_timer_range = (20, 40)
        else: # Stage 3 and beyond
            self.base_asteroid_speed = 7.0
            self.asteroid_spawn_timer_range = (15, 30)
        
        self.asteroid_spawn_timer = self.np_random.integers(
            self.asteroid_spawn_timer_range[0], self.asteroid_spawn_timer_range[1]
        )

    def _update_player(self):
        """Applies physics to the player."""
        # Apply gravity
        self.player_vy += self.GRAVITY
        self.player_y += self.player_vy

        # Check for ground collision
        if self.player_y + self.PLAYER_HEIGHT >= self.ground_y:
            if not self.on_ground:
                # Sound: Land
                self._create_particles(self.player_rect.midbottom, 10, -1) # Landing dust
            self.player_y = self.ground_y - self.PLAYER_HEIGHT
            self.player_vy = 0
            self.on_ground = True
        
        self.player_rect = pygame.Rect(self.PLAYER_X, self.player_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

    def _update_asteroids(self):
        """Moves asteroids and removes them if they go off-screen."""
        for asteroid in self.asteroids[:]:
            asteroid['x'] -= asteroid['speed']
            asteroid['rect'].x = int(asteroid['x'])
            if asteroid['rect'].right < 0:
                self.asteroids.remove(asteroid)

    def _handle_spawning(self):
        """Handles the timer-based spawning of new asteroids."""
        self.asteroid_spawn_timer -= 1
        if self.asteroid_spawn_timer <= 0:
            self._spawn_asteroid()
            self.asteroid_spawn_timer = self.np_random.integers(
                self.asteroid_spawn_timer_range[0], self.asteroid_spawn_timer_range[1]
            )

    def _spawn_asteroid(self):
        """Creates a new asteroid and adds it to the game."""
        size = self.np_random.integers(20, 50)
        # Spawn height avoids being too close to the ground
        min_y = 50
        max_y = self.ground_y - size - 20 # -20 provides a small gap
        y = self.np_random.integers(min_y, max_y)
        
        speed_variance = self.np_random.uniform(-0.5, 1.5)
        speed = self.base_asteroid_speed + speed_variance

        asteroid = {
            'x': float(self.SCREEN_WIDTH + size),
            'rect': pygame.Rect(self.SCREEN_WIDTH + size, y, size, size),
            'speed': speed,
            'size': size,
            'color': random.choice(self.COLOR_ASTEROID),
        }
        self.asteroids.append(asteroid)

    def _check_collisions(self):
        """Checks for collisions between the player and asteroids."""
        player_hitbox = self.player_rect.inflate(-4, -4) # Make hitbox slightly smaller
        for asteroid in self.asteroids:
            if player_hitbox.colliderect(asteroid['rect']):
                # More precise circle-rect collision
                circle_x, circle_y, r = asteroid['rect'].centerx, asteroid['rect'].centery, asteroid['size'] / 2
                
                # Find the closest point on the rect to the circle's center
                closest_x = max(player_hitbox.left, min(circle_x, player_hitbox.right))
                closest_y = max(player_hitbox.top, min(circle_y, player_hitbox.bottom))

                distance_sq = (circle_x - closest_x)**2 + (circle_y - closest_y)**2
                if distance_sq < (r**2):
                    self._create_particles(player_hitbox.center, 50, 0, (255, 50, 50))
                    return True
        return False
        
    def _create_particles(self, pos, count, base_vy, color=None):
        """Creates a burst of particles at a given position."""
        if color is None:
            color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed + base_vy]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color,
            })

    def _update_particles(self):
        """Updates particle positions, fades them, and removes dead ones."""
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
        }

    def _get_observation(self):
        # Render all game elements
        self._render_frame()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_frame(self):
        """Renders all visual elements for a single frame."""
        # Background
        self.screen.fill(self.COLOR_BG)
        for x, y in self.stars:
            self.screen.set_at((x, y), self.COLOR_STAR)
        
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.ground_y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.ground_y))

        # Asteroids
        for asteroid in self.asteroids:
            pygame.gfxdraw.filled_circle(
                self.screen, int(asteroid['rect'].centerx), int(asteroid['rect'].centery),
                int(asteroid['size'] / 2), asteroid['color']
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(asteroid['rect'].centerx), int(asteroid['rect'].centery),
                int(asteroid['size'] / 2), asteroid['color']
            )

        # Player squash and stretch animation
        squash = min(5, max(0, self.player_vy))
        stretch = min(10, max(0, -self.player_vy * 0.8))
        
        player_render_rect = pygame.Rect(
            self.player_rect.x - squash / 2,
            self.player_rect.y - stretch,
            self.player_rect.width + squash,
            self.player_rect.height + stretch
        )
        
        # Player Glow
        glow_surface = pygame.Surface((player_render_rect.width * 2, player_render_rect.height * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surface, self.COLOR_PLAYER_GLOW, glow_surface.get_rect())
        self.screen.blit(glow_surface, (player_render_rect.centerx - glow_surface.get_width() / 2, player_render_rect.centery - glow_surface.get_height() / 2), special_flags=pygame.BLEND_RGBA_ADD)

        # Player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_render_rect, border_radius=4)
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 3, 3))
            self.screen.blit(temp_surf, (int(p['pos'][0]), int(p['pos'][1])))
            
        # UI
        self._render_ui()

    def _render_ui(self):
        """Renders the UI text on top of the game."""
        # Stage Text
        stage_text = self.font_ui.render(f"STAGE: {self.current_stage}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Timer Text
        time_left = max(0, self.total_game_time_seconds - (self.steps / self.FPS))
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Score Text
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))
        
        # Game Over Text
        if self.game_over:
            if self.steps >= self.MAX_STEPS:
                end_msg = "YOU WIN!"
            else:
                end_msg = "GAME OVER"
            
            game_over_surf = self.font_game_over.render(end_msg, True, self.COLOR_UI_TEXT)
            self.screen.blit(game_over_surf, (self.SCREEN_WIDTH // 2 - game_over_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - game_over_surf.get_height() // 2))

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


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Space Hopper")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Get player input
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]
        
        action = [0, 0, 0] # Default no-op
        action[1] = 1 if space_held else 0
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Episode finished. Final Score: {info['score']:.2f}. Press 'R' to restart.")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(GameEnv.FPS)
        
    env.close()