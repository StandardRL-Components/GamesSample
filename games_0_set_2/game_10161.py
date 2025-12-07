import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:56:02.901889
# Source Brief: brief_00161.md
# Brief Index: 161
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for game objects to keep code clean
class GameObject:
    """A simple class to hold state for stones."""
    def __init__(self, x, y, radius, color):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.radius = radius
        self.color = color

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player manipulates localized gravity 
    to guide a river stone through a tidal channel to the ocean, racing an AI opponent.
    This environment prioritizes visual quality and fluid, real-time gameplay.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manipulate localized gravity to guide your stone down a tidal channel to the ocean, racing against an AI opponent."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move your gravity well. Press space to boost the gravitational pull."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CHANNEL_TOP = 50
        self.CHANNEL_BOTTOM = 350
        self.OCEAN_X = self.SCREEN_WIDTH - 20

        # Game constants
        self.MAX_STEPS = 2000
        self.GRAVITY_RADIUS = 150
        self.GRAVITY_STRENGTH_BASE = 0.08
        self.PLAYER_GRAVITY_SPEED = 4.0
        self.STONE_MASS = 5.0
        self.STONE_DAMPING = 0.98
        self.AI_SPEED = 2.0
        
        # Colors
        self.COLOR_BG = (15, 23, 42) # Dark Slate Blue
        self.COLOR_CHANNEL = (30, 41, 59) # Darker Slate
        self.COLOR_OCEAN = (56, 189, 248) # Sky Blue
        self.COLOR_PLAYER_STONE = (234, 179, 8) # Amber
        self.COLOR_PLAYER_GRAVITY = (74, 222, 128) # Green
        self.COLOR_OPPONENT_STONE = (156, 163, 175) # Gray
        self.COLOR_OPPONENT_GRAVITY = (168, 85, 247) # Purple
        self.COLOR_UI = (241, 245, 249) # Off-white
        self.COLOR_PARTICLE = (100, 116, 139)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Persistent state (as per brief)
        self.current_strength_base = 0.5
        self.tidal_freq_base = 0.05
        
        # Initialize state variables to be defined in reset()
        self.player_stone = None
        self.opponent_stone = None
        self.player_gravity = None
        self.opponent_gravity = None
        self.water_particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_player_dist_to_ocean = 0
        self.current_strength = 0
        self.tidal_freq = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        start_y_player = self.np_random.uniform(self.CHANNEL_TOP + 20, self.CHANNEL_BOTTOM - 20)
        start_y_opponent = self.np_random.uniform(self.CHANNEL_TOP + 20, self.CHANNEL_BOTTOM - 20)

        self.player_stone = GameObject(50, start_y_player, 12, self.COLOR_PLAYER_STONE)
        self.opponent_stone = GameObject(50, start_y_opponent, 12, self.COLOR_OPPONENT_STONE)
        
        self.player_gravity = pygame.Vector2(self.player_stone.pos)
        self.opponent_gravity = pygame.Vector2(self.opponent_stone.pos)
        
        self.current_strength = self.current_strength_base
        self.tidal_freq = self.tidal_freq_base

        self.last_player_dist_to_ocean = self.SCREEN_WIDTH - self.player_stone.pos.x

        self.water_particles = []
        for _ in range(150):
            self.water_particles.append(pygame.Vector2(
                self.np_random.uniform(0, self.SCREEN_WIDTH),
                self.np_random.uniform(self.CHANNEL_TOP, self.CHANNEL_BOTTOM)
            ))

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        self.steps += 1

        # 1. Update Player Gravity Well from action
        if movement == 1: # Up
            self.player_gravity.y -= self.PLAYER_GRAVITY_SPEED
        elif movement == 2: # Down
            self.player_gravity.y += self.PLAYER_GRAVITY_SPEED
        elif movement == 3: # Left
            self.player_gravity.x -= self.PLAYER_GRAVITY_SPEED
        elif movement == 4: # Right
            self.player_gravity.x += self.PLAYER_GRAVITY_SPEED
        
        # Clamp gravity well to screen
        self.player_gravity.x = max(0, min(self.SCREEN_WIDTH, self.player_gravity.x))
        self.player_gravity.y = max(0, min(self.SCREEN_HEIGHT, self.player_gravity.y))

        # 2. Run Opponent AI
        self._run_opponent_ai()

        # 3. Update game difficulty
        self.current_strength = self.current_strength_base + 0.2 * (self.steps // 500)
        self.tidal_freq = self.tidal_freq_base + 0.01 * (self.steps // 1000)

        # 4. Update Physics
        self._update_physics(space_held)

        # 5. Calculate Reward & Check Termination
        reward = self._calculate_reward()
        terminated = self._check_termination() # This also updates terminal score
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated or truncated:
            self.game_over = True
            if self.player_stone.pos.x >= self.OCEAN_X and self.opponent_stone.pos.x < self.OCEAN_X:
                 self.score = 100
                 reward = 100
            elif self.opponent_stone.pos.x >= self.OCEAN_X and self.player_stone.pos.x < self.OCEAN_X:
                 self.score = -100
                 reward = -100
            elif self.player_stone.pos.x >= self.OCEAN_X and self.opponent_stone.pos.x >= self.OCEAN_X:
                if self.player_stone.pos.x > self.opponent_stone.pos.x:
                    self.score = 100
                    reward = 100
                else:
                    self.score = -100
                    reward = -100
            elif truncated and not terminated: # Time ran out
                self.score = 0
                reward = 0
        else:
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _run_opponent_ai(self):
        # AI targets a point ahead and towards the center of its stone
        target_x = self.opponent_stone.pos.x + 80
        target_y = self.SCREEN_HEIGHT / 2
        target = pygame.Vector2(target_x, target_y)

        direction = (target - self.opponent_gravity).normalize() if (target - self.opponent_gravity).length() > 0 else pygame.Vector2(0,0)
        self.opponent_gravity += direction * self.AI_SPEED

        # Clamp opponent gravity well
        self.opponent_gravity.x = max(0, min(self.SCREEN_WIDTH, self.opponent_gravity.x))
        self.opponent_gravity.y = max(0, min(self.SCREEN_HEIGHT, self.opponent_gravity.y))

    def _update_physics(self, player_boost):
        # Update water particles for visual effect
        for p in self.water_particles:
            p.x += self.current_strength + math.sin(self.steps * self.tidal_freq * 2 * math.pi) * 2.0
            if p.x > self.SCREEN_WIDTH:
                p.x = 0
                p.y = self.np_random.uniform(self.CHANNEL_TOP, self.CHANNEL_BOTTOM)

        # Calculate forces for both stones
        stones = [self.player_stone, self.opponent_stone]
        for stone in stones:
            # a. Current Force
            tidal_force = math.sin(self.steps * self.tidal_freq * 2 * math.pi) * self.current_strength * 2.5
            force_current = pygame.Vector2(self.current_strength + tidal_force, 0)
            
            # b. Player Gravity Force
            player_g_strength = self.GRAVITY_STRENGTH_BASE * 2 if player_boost else self.GRAVITY_STRENGTH_BASE
            force_player_g = self._calculate_gravity_force(stone.pos, self.player_gravity, player_g_strength)
            
            # c. Opponent Gravity Force
            force_opponent_g = self._calculate_gravity_force(stone.pos, self.opponent_gravity, self.GRAVITY_STRENGTH_BASE)

            # d. Total force
            total_force = force_current + force_player_g + force_opponent_g

            # e. Update velocity and position (Euler integration)
            acceleration = total_force / self.STONE_MASS
            stone.vel = stone.vel * self.STONE_DAMPING + acceleration
            stone.pos += stone.vel
            
            # f. Boundary collision
            if stone.pos.y - stone.radius < self.CHANNEL_TOP:
                stone.pos.y = self.CHANNEL_TOP + stone.radius
                stone.vel.y *= -0.5
            if stone.pos.y + stone.radius > self.CHANNEL_BOTTOM:
                stone.pos.y = self.CHANNEL_BOTTOM - stone.radius
                stone.vel.y *= -0.5
            
            stone.pos.x = max(stone.radius, min(self.SCREEN_WIDTH - stone.radius, stone.pos.x))

    def _calculate_gravity_force(self, stone_pos, gravity_pos, strength):
        dist_vec = gravity_pos - stone_pos
        distance = dist_vec.length()
        
        if distance < 1 or distance > self.GRAVITY_RADIUS:
            return pygame.Vector2(0, 0)
        
        falloff = 1.0 - (distance / self.GRAVITY_RADIUS)
        force_magnitude = strength * falloff * falloff
        
        direction = dist_vec.normalize()
        return direction * force_magnitude

    def _calculate_reward(self):
        current_dist = self.SCREEN_WIDTH - self.player_stone.pos.x
        reward = (self.last_player_dist_to_ocean - current_dist) * 0.1
        self.last_player_dist_to_ocean = current_dist
        return reward
    
    def _check_termination(self):
        player_wins = self.player_stone.pos.x >= self.OCEAN_X
        opponent_wins = self.opponent_stone.pos.x >= self.OCEAN_X
        return player_wins or opponent_wins

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw channel and ocean
        pygame.draw.rect(self.screen, self.COLOR_CHANNEL, (0, self.CHANNEL_TOP, self.SCREEN_WIDTH, self.CHANNEL_BOTTOM - self.CHANNEL_TOP))
        pygame.draw.rect(self.screen, self.COLOR_OCEAN, (self.OCEAN_X, self.CHANNEL_TOP, self.SCREEN_WIDTH - self.OCEAN_X, self.CHANNEL_BOTTOM - self.CHANNEL_TOP))
        
        # Draw water particles
        for p in self.water_particles:
            pygame.draw.line(self.screen, self.COLOR_PARTICLE, p, p + pygame.Vector2(3,0), 1)

        # Draw gravity wells
        self._draw_radial_gradient(self.player_gravity, self.COLOR_PLAYER_GRAVITY, self.GRAVITY_RADIUS)
        self._draw_radial_gradient(self.opponent_gravity, self.COLOR_OPPONENT_GRAVITY, self.GRAVITY_RADIUS)
        
        # Draw stones
        self._draw_stone(self.opponent_stone)
        self._draw_stone(self.player_stone)

    def _draw_radial_gradient(self, pos, color, radius):
        max_alpha = 70
        for i in range(radius, 0, -3):
            alpha = int(max_alpha * (1 - (i / radius))**2)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), i, (*color, alpha))

    def _draw_stone(self, stone):
        pygame.draw.circle(self.screen, stone.color, (int(stone.pos.x), int(stone.pos.y)), stone.radius)
        highlight_pos = stone.pos + pygame.Vector2(-3, -3)
        highlight_color = tuple(min(255, c + 50) for c in stone.color)
        pygame.draw.circle(self.screen, highlight_color, (int(highlight_pos.x), int(highlight_pos.y)), stone.radius // 2)

    def _render_ui(self):
        score_text = self.font.render(f"Score: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI)
        steps_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(steps_text, steps_rect)

        current_text = self.small_font.render(f"Current: {self.current_strength:.2f}", True, self.COLOR_UI)
        self.screen.blit(current_text, (10, self.SCREEN_HEIGHT - 30))
        
        p1_indicator = self.small_font.render("P1", True, self.COLOR_PLAYER_GRAVITY)
        self.screen.blit(p1_indicator, self.player_gravity + pygame.Vector2(10, 10))
        
        p2_indicator = self.small_font.render("P2", True, self.COLOR_OPPONENT_GRAVITY)
        self.screen.blit(p2_indicator, self.opponent_gravity + pygame.Vector2(10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": (self.player_stone.pos.x, self.player_stone.pos.y),
            "opponent_pos": (self.opponent_stone.pos.x, self.opponent_stone.pos.y),
            "current_strength": self.current_strength,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be run by the autograder
    
    # Switch to a visible driver for interactive play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tidal Channel")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # 0=none
        space_held = 0 # 0=released
        shift_held = 0 # 0=released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            
        clock.tick(30) # Run at 30 FPS

    env.close()