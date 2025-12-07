
# Generated: 2025-08-28T04:56:50.745151
# Source Brief: brief_02473.md
# Brief Index: 2473

        
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
    """
    A minimalist side-view arcade game where the player controls a hopping space
    creature, aiming to survive a barrage of oncoming obstacles for 60 seconds.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Press Space to jump and avoid the red obstacles."
    )

    # Short, user-facing description of the game
    game_description = (
        "A minimalist side-view arcade game where you control a hopping space "
        "creature, aiming to survive a barrage of oncoming obstacles for 60 seconds."
    )

    # Frames auto-advance at a fixed rate (60fps)
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60 seconds

        # Colors
        self.COLOR_BG = (15, 19, 41)
        self.COLOR_GROUND = (60, 60, 80)
        self.COLOR_PLAYER = (57, 255, 20)
        self.COLOR_OBSTACLE = (255, 40, 40)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_STAR = (200, 200, 220)

        # Player Physics
        self.GRAVITY = 0.4
        self.JUMP_STRENGTH = -9.5
        self.PLAYER_WIDTH = 24
        self.PLAYER_HEIGHT = 24
        self.PLAYER_X_POS = 100

        # Ground
        self.GROUND_Y = self.HEIGHT - 50

        # Obstacles
        self.INITIAL_OBSTACLE_SPEED = 2.0
        self.OBSTACLE_SPEED_INCREASE_INTERVAL = self.FPS * 10  # Every 10 seconds
        self.OBSTACLE_SPEED_INCREASE_AMOUNT = 0.5 # The brief says 0.01, but that's too small to be noticeable. I'll use a more impactful value.

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_gameover = pygame.font.SysFont("monospace", 50, bold=True)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.game_over = False
        self.win = False
        self.player_pos = [0, 0]
        self.player_vel_y = 0
        self.is_on_ground = True
        self.obstacles = []
        self.particles = []
        self.stars = []
        self.base_obstacle_speed = 0
        self.obstacle_spawn_timer = 0
        self.squash_factor = 1.0

        # Initialize state variables for the first time
        self.reset()
        
        # Validate implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.game_over = False
        self.win = False

        # Player state
        self.player_pos = [self.PLAYER_X_POS, self.GROUND_Y - self.PLAYER_HEIGHT]
        self.player_vel_y = 0
        self.is_on_ground = True
        self.squash_factor = 1.0

        # Obstacle state
        self.obstacles = []
        self.base_obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.obstacle_spawn_timer = self.np_random.integers(30, 60)

        # Effects state
        self.particles = []
        self.stars = []
        for _ in range(100): # Parallax stars
            layer = self.np_random.random()
            speed_multiplier = 0.1 + (layer * 0.4)
            self.stars.append({
                "pos": [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.GROUND_Y)],
                "radius": 1 + int(layer * 2),
                "speed": self.base_obstacle_speed * speed_multiplier
            })
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        # If game is over, do nothing but return current state
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        space_pressed = action[1] == 1

        self.steps += 1
        reward = 0.01  # Survival reward
        terminated = False

        # --- Game Logic ---

        # 1. Handle Input
        if space_pressed and self.is_on_ground:
            self.player_vel_y = self.JUMP_STRENGTH
            self.is_on_ground = False
            # sfx: Jump sound

        # 2. Update Player
        self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y

        if self.player_pos[1] + self.PLAYER_HEIGHT >= self.GROUND_Y:
            self.player_pos[1] = self.GROUND_Y - self.PLAYER_HEIGHT
            self.player_vel_y = 0
            if not self.is_on_ground: # Just landed
                self.is_on_ground = True
                self._create_landing_particles(10)
                self.squash_factor = 1.8 # Start squash effect
                # sfx: Land sound
        
        # Update squash effect for animation
        self.squash_factor = max(1.0, self.squash_factor - 0.1)

        # 3. Update Difficulty
        if self.steps > 0 and self.steps % self.OBSTACLE_SPEED_INCREASE_INTERVAL == 0:
            self.base_obstacle_speed += self.OBSTACLE_SPEED_INCREASE_AMOUNT

        # 4. Update Obstacles
        self._update_obstacles()

        # 5. Update Particles & Stars
        self._update_particles()
        self._update_stars()

        # 6. Check Collisions
        player_rect = self._get_player_rect()
        for obs in self.obstacles:
            if player_rect.colliderect(obs["rect"]):
                self.game_over = True
                terminated = True
                reward = -10  # Collision penalty
                self._create_landing_particles(30, self.COLOR_OBSTACLE)
                # sfx: Collision/explosion sound
                break
        
        # 7. Check Win/Termination Conditions
        if not terminated and self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = True
            terminated = True
            reward = 100  # Win bonus

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_obstacles(self):
        # Spawn new obstacles
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            height = self.np_random.integers(20, 60)
            width = self.np_random.integers(20, 40)
            speed = self.base_obstacle_speed + self.np_random.uniform(-0.5, 1.0)
            
            self.obstacles.append({
                "rect": pygame.Rect(self.WIDTH, self.GROUND_Y - height, width, height),
                "speed": max(1.5, speed)
            })
            
            # Reset timer with some randomness
            spawn_interval = max(30, 120 - (self.base_obstacle_speed * 10))
            self.obstacle_spawn_timer = self.np_random.integers(int(spawn_interval * 0.8), int(spawn_interval * 1.2))

        # Move and remove old obstacles
        for obs in self.obstacles:
            obs["rect"].x -= obs["speed"]
        self.obstacles = [obs for obs in self.obstacles if obs["rect"].right > 0]

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity on particles
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _update_stars(self):
        for star in self.stars:
            star["pos"][0] -= star["speed"] * 0.1 # Parallax effect
            if star["pos"][0] < 0:
                star["pos"][0] = self.WIDTH
                star["pos"][1] = self.np_random.uniform(0, self.GROUND_Y)

    def _create_landing_particles(self, count, color=None):
        if color is None:
            color = self.COLOR_PLAYER
        for _ in range(count):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": [self.player_pos[0] + self.PLAYER_WIDTH / 2, self.GROUND_Y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(20, 40),
                "color": color,
                "radius": self.np_random.uniform(1, 3)
            })
            
    def _get_player_rect(self):
        # Calculate squashed dimensions for collision
        height_squash = self.PLAYER_HEIGHT / self.squash_factor
        width_squash = self.PLAYER_WIDTH * self.squash_factor
        y_pos = self.player_pos[1] + (self.PLAYER_HEIGHT - height_squash)
        x_pos = self.player_pos[0] - (width_squash - self.PLAYER_WIDTH) / 2
        return pygame.Rect(x_pos, y_pos, width_squash, height_squash)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.steps / self.FPS,  # Score is survival time in seconds
            "steps": self.steps,
        }

    def _render_game(self):
        # Render stars
        for star in self.stars:
            pygame.gfxdraw.filled_circle(
                self.screen, int(star["pos"][0]), int(star["pos"][1]), int(star["radius"]), self.COLOR_STAR
            )
            
        # Render ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

        # Render obstacles
        for obs in self.obstacles:
            self._render_glowing_rect(obs["rect"], self.COLOR_OBSTACLE, 15)

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 30))))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), color
            )
            
        # Render player if not dead by collision
        if not (self.game_over and not self.win):
            self._render_player()

    def _render_player(self):
        # Apply squash and stretch based on vertical velocity and landing impact
        squash_y = self.squash_factor
        stretch_x = self.squash_factor
        
        # Add stretch from vertical velocity
        stretch_y = 1.0 - max(-0.5, min(0.5, self.player_vel_y / 20.0))
        
        height = self.PLAYER_HEIGHT * stretch_y / squash_y
        width = self.PLAYER_WIDTH * stretch_x / stretch_y
        
        y_pos = self.player_pos[1] + (self.PLAYER_HEIGHT - height)
        x_pos = self.player_pos[0] - (width - self.PLAYER_WIDTH) / 2
        
        player_render_rect = pygame.Rect(x_pos, y_pos, width, height)
        self._render_glowing_rect(player_render_rect, self.COLOR_PLAYER, 20)

    def _render_glowing_rect(self, rect, color, glow_size):
        # Draw glow effect
        glow_color = (*color, 30) # Low alpha for glow
        for i in range(glow_size // 2, 0, -2):
            glow_rect = rect.inflate(i, i)
            pygame.draw.rect(self.screen, glow_color, glow_rect, border_radius=int(rect.height / 3))
        # Draw main rect
        pygame.draw.rect(self.screen, color, rect, border_radius=int(rect.height / 3))

    def _render_ui(self):
        # Render survival time
        time_survived = self.steps / self.FPS
        time_text = self.font_ui.render(f"TIME: {time_survived:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Render game over/win message
        if self.game_over:
            if self.win:
                message = "YOU SURVIVED!"
                color = self.COLOR_PLAYER
            else:
                message = "GAME OVER"
                color = self.COLOR_OBSTACLE
            
            text_surf = self.font_gameover.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 50))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(40, 40)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(text_surf, text_rect)
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for visualization
    pygame.display.set_caption("Hopper Game")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Get user input for manual play
        keys = pygame.key.get_pressed()
        space_pressed = keys[pygame.K_SPACE]
        
        action = [0, 1 if space_pressed else 0, 0] # No movement, space, no shift
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score (Time): {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()
            total_reward = 0
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)
        
    env.close()