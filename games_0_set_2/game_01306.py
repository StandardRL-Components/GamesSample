import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Hold [SPACE] to charge a jump, release to leap. Higher charge means a higher jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Reach the top of a procedural tower by mastering a variable-height jump. Avoid the red obstacles!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Colors
        self.COLOR_BG_TOP = (4, 12, 48)
        self.COLOR_BG_BOTTOM = (24, 48, 112)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_OBSTACLE = (255, 64, 64)
        self.COLOR_OBSTACLE_OUTLINE = (128, 32, 32)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_CHARGE_BAR = (255, 255, 0)
        self.COLOR_CHARGE_BAR_BG = (100, 100, 100, 128)

        # Game constants
        self.GRAVITY = 0.4
        self.PLAYER_SIZE = 20
        self.MAX_JUMP_CHARGE = 30  # frames to hold for max charge
        self.MIN_JUMP_VEL = 8
        self.MAX_JUMP_VEL = 15
        self.SCROLL_THRESHOLD = self.HEIGHT / 2.5
        self.PIXELS_PER_UNIT = 10.0
        self.MAX_EPISODE_STEPS = 1500
        self.WIN_HEIGHT = 100

        # State variables will be initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.obstacles = None
        self.particles = None
        self.is_grounded = None
        self.jump_charge = None
        self.was_space_held = None
        self.height = None
        self.max_height_achieved = None
        self.steps = None
        self.game_over = None
        self.last_landed_platform_gap = None
        
        # Initialize state variables
        # self.reset() # This is called by the test harness, no need to call in init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.height = 0
        self.max_height_achieved = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = True
        self.jump_charge = 0
        self.was_space_held = False
        
        self.obstacles = deque()
        self.particles = deque()
        
        # Create initial platform
        initial_platform = pygame.Rect(0, self.HEIGHT - 20, self.WIDTH, 20)
        self.obstacles.append(initial_platform)
        
        # Pre-generate platforms
        last_y = initial_platform.y
        while len(self.obstacles) < 15:
            last_y = self._generate_obstacle(last_y)

        return self._get_observation(), self._get_info()

    def _generate_obstacle(self, last_y):
        # Difficulty scaling
        difficulty_factor = min(1.0, self.height / self.WIN_HEIGHT)
        
        min_gap_y = 60 - 30 * difficulty_factor
        max_gap_y = 120 - 60 * difficulty_factor
        
        min_gap_x = self.PLAYER_SIZE * 3.0 - (self.PLAYER_SIZE * 1.5 * difficulty_factor)
        max_gap_x = self.PLAYER_SIZE * 5.0 - (self.PLAYER_SIZE * 2.0 * difficulty_factor)
        
        width = self.np_random.integers(int(self.PLAYER_SIZE * 1.5), int(self.WIDTH * 0.4))
        
        gap_y = self.np_random.uniform(min_gap_y, max_gap_y)
        new_y = last_y - gap_y - 20

        side = self.np_random.choice(['left', 'right'])
        if side == 'left':
            x = self.np_random.uniform(-width * 0.5, self.WIDTH - max_gap_x - width)
        else:
            x = self.np_random.uniform(max_gap_x, self.WIDTH - width * 0.5)
            
        x = np.clip(x, 0, self.WIDTH - width)

        obstacle = pygame.Rect(int(x), int(new_y), int(width), 20)
        self.obstacles.appendleft(obstacle)
        return new_y

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # unused
        space_held = action[1] == 1
        shift_held = action[2] == 1 # unused

        reward = 0
        
        # --- Handle Input & Jump Mechanics ---
        self._handle_input(space_held)

        # --- Update Physics & World Scroll ---
        scroll_amount = self._update_physics_and_scroll()
        
        # --- Update Height & Height Reward ---
        height_increase = scroll_amount / self.PIXELS_PER_UNIT
        if height_increase > 0:
            self.height += height_increase
            if self.height > self.max_height_achieved:
                reward += (self.height - self.max_height_achieved) * 0.1
                self.max_height_achieved = self.height

        # --- Collision Detection ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- Update Obstacles ---
        self._update_obstacles()
        
        # --- Update Particles ---
        self._update_particles()
        
        # --- Check Termination Conditions ---
        terminated = False
        if self.player_pos.y > self.HEIGHT: # Fell off screen
            self.game_over = True
        
        if self.height >= self.WIN_HEIGHT:
            self.game_over = True
            reward += 100.0  # Win bonus
        
        if self.game_over and not (self.height >= self.WIN_HEIGHT):
            reward -= 10.0 # Collision penalty
        
        self.steps += 1
        truncated = False
        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            self.game_over = True # End the game if truncated

        terminated = self.game_over and not truncated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, space_held):
        if space_held and self.is_grounded:
            self.jump_charge = min(self.MAX_JUMP_CHARGE, self.jump_charge + 1)
        
        # Jump on release
        if not space_held and self.was_space_held and self.is_grounded and self.jump_charge > 0:
            charge_ratio = self.jump_charge / self.MAX_JUMP_CHARGE
            jump_velocity = self.MIN_JUMP_VEL + (self.MAX_JUMP_VEL - self.MIN_JUMP_VEL) * charge_ratio
            self.player_vel.y = -jump_velocity
            self.is_grounded = False
            self.jump_charge = 0
            # SFX: Jump sound
            self._create_jump_particles(20)

        self.was_space_held = space_held
        
    def _update_physics_and_scroll(self):
        # Apply gravity
        if not self.is_grounded:
            self.player_vel.y += self.GRAVITY
        
        # Update position
        self.player_pos += self.player_vel

        # World scrolling
        scroll = 0
        if self.player_pos.y < self.SCROLL_THRESHOLD:
            scroll = self.SCROLL_THRESHOLD - self.player_pos.y
            self.player_pos.y = self.SCROLL_THRESHOLD
            for obstacle in self.obstacles:
                obstacle.y += scroll
            for particle in self.particles:
                particle['pos'].y += scroll
        return scroll

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, 
                                  self.player_pos.y - self.PLAYER_SIZE, 
                                  self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        self.is_grounded = False
        reward = 0
        
        for obstacle in self.obstacles:
            if player_rect.colliderect(obstacle):
                # Check for landing on top
                if self.player_vel.y > 0 and player_rect.bottom < obstacle.top + (self.PLAYER_SIZE/2):
                    self.player_pos.y = obstacle.top
                    self.player_vel.y = 0
                    self.is_grounded = True
                    self.jump_charge = 0
                    # SFX: Land sound
                    self._create_land_particles(10)
                    
                    # Reward for landing on smaller platforms
                    max_w = self.WIDTH * 0.4
                    min_w = self.PLAYER_SIZE * 1.5
                    normalized_width = (obstacle.width - min_w) / (max_w - min_w)
                    reward += (1.0 - normalized_width) * 0.5 # Max +0.5 for smallest platform

                    break
                else: # Collision with side or bottom
                    self.game_over = True
                    # SFX: Crash sound
                    self._create_crash_particles(50)
                    break
        return reward

    def _update_obstacles(self):
        # Remove obstacles that are off-screen
        while self.obstacles and self.obstacles[-1].top > self.HEIGHT:
            self.obstacles.pop()
        
        # Generate new obstacles if needed
        if self.obstacles:
            top_y = self.obstacles[0].y
            if top_y > -20:
                self._generate_obstacle(top_y)

    def _create_jump_particles(self, count):
        for _ in range(count):
            angle = self.np_random.uniform(math.pi * 0.9, math.pi * 2.1)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': self.player_pos.copy() + pygame.Vector2(0, -self.PLAYER_SIZE/2),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': self.COLOR_PLAYER_GLOW
            })

    def _create_land_particles(self, count):
        for _ in range(count):
            angle = self.np_random.uniform(-math.pi * 0.9, -math.pi * 0.1)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': self.player_pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 25),
                'color': self.COLOR_PLAYER
            })

    def _create_crash_particles(self, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': self.player_pos.copy() - pygame.Vector2(0, self.PLAYER_SIZE/2),
                'vel': vel,
                'life': self.np_random.integers(30, 60),
                'color': random.choice([self.COLOR_PLAYER, self.COLOR_OBSTACLE, self.COLOR_PLAYER_GLOW])
            })
            
    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw gradient background
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, obstacle, 2)
            
        # Draw particles
        for p in self.particles:
            size = max(1, int(p['life'] / 5))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), size)

        # Draw player if not game over
        if not self.game_over or self.height >= self.WIN_HEIGHT:
            player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
            player_rect.center = (self.player_pos.x, self.player_pos.y - self.PLAYER_SIZE / 2)
            
            # Glow effect
            glow_rect = player_rect.inflate(self.PLAYER_SIZE * 0.5, self.PLAYER_SIZE * 0.5)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, self.COLOR_PLAYER_GLOW + (80,), glow_surface.get_rect(), border_radius=5)
            self.screen.blit(glow_surface, glow_rect.topleft)

            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Draw jump charge bar
        if self.is_grounded and self.jump_charge > 0:
            charge_ratio = self.jump_charge / self.MAX_JUMP_CHARGE
            bar_width = 40
            bar_height = 8
            
            bg_rect = pygame.Rect(0, 0, bar_width, bar_height)
            bg_rect.center = (self.player_pos.x, self.player_pos.y + 15)
            
            fill_width = int(bar_width * charge_ratio)
            fill_rect = pygame.Rect(bg_rect.x, bg_rect.y, fill_width, bar_height)
            
            # Use a surface for alpha blending
            s = pygame.Surface((bar_width, bar_height), pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_CHARGE_BAR_BG, (0, 0, bar_width, bar_height), border_radius=4)
            self.screen.blit(s, bg_rect.topleft)
            pygame.draw.rect(self.screen, self.COLOR_CHARGE_BAR, fill_rect, border_radius=4)


    def _render_ui(self):
        height_text = f"Height: {self.height:.1f} / {self.WIN_HEIGHT}"
        text_surface = self.font.render(height_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))
        
        if self.game_over:
            if self.height >= self.WIN_HEIGHT:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text_surface = self.font.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.height,
            "steps": self.steps,
            "max_height": self.max_height_achieved,
            "is_grounded": self.is_grounded,
            "jump_charge": self.jump_charge,
        }

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
        # Need to reset first to initialize everything
        self.reset()
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # The validation call in __init__ was removed, let's call it here for the main script
    # env_for_validation = GameEnv()
    # env_for_validation.validate_implementation()
    # env_for_validation.close()

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Human Play Loop ---
    # This loop allows a human to play the game.
    # It demonstrates the environment's functionality.
    
    # Use a dictionary to track held keys for smooth controls
    keys_held = {
        'up': False, 'down': False, 'left': False, 'right': False,
        'space': False, 'shift': False
    }

    print("\n" + "="*30)
    print(" VERTICAL JUMPER - HUMAN TEST")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30)

    total_reward = 0
    
    # Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Vertical Jumper")
    
    while not done:
        # --- Event Handling (for human input) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: keys_held['space'] = True
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_held['shift'] = True
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    total_reward = 0
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: keys_held['space'] = False
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_held['shift'] = False

        # --- Action Mapping ---
        # Movement is unused in this game, so it's always 0 (none)
        movement_action = 0 
        space_action = 1 if keys_held['space'] else 0
        shift_action = 1 if keys_held['shift'] else 0
        
        action = [movement_action, space_action, shift_action]

        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True
        
        total_reward += reward
        
        # --- Render to the Display Window ---
        # The observation is already a rendered frame, so we just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Control Framerate ---
        env.clock.tick(60) # Run at 60 FPS for smooth human play

    print(f"Game Over! Final Score (Height): {info['score']:.2f}, Total Reward: {total_reward:.2f}")
    env.close()