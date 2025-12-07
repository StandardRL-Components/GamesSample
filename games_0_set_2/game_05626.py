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
        "Controls: ←→ to jump sideways. ↑ for a medium vertical jump. "
        "Hold Space for a high jump. Hold Shift for a short hop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between procedurally generated platforms to reach the top. "
        "Race against the clock and don't fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 10)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_INNER = (150, 255, 150)
    COLOR_PLATFORM = (240, 240, 240)
    COLOR_PLATFORM_OUTLINE = (100, 100, 100)
    COLOR_PARTICLE = (255, 255, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TOP_PLATFORM = (255, 223, 0) # Gold

    # Physics
    GRAVITY = 0.4
    JUMP_HIGH = -11
    JUMP_MEDIUM = -9
    JUMP_LOW = -7
    JUMP_SIDE_VEL = 6
    MAX_VEL_Y = 15

    # Game
    FPS = 30
    MAX_STEPS = 1000
    TIMER_SECONDS = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        # Set dummy video driver for headless operation
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables to None, to be set in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.timer = None
        self.player_pos = None
        self.player_vel = None
        self.player_size = None
        self.player_rect = None
        self.on_platform = None
        self.platforms = None
        self.top_platform_index = None
        self.current_platform_index = None
        self.last_platform_index = None
        self.particles = None
        
        # Call reset to initialize the game state
        # self.reset() # Not needed as it's typically called by the training loop first

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.TIMER_SECONDS * self.FPS
        
        self.player_size = 20
        self.player_vel = pygame.Vector2(0, 0)
        self.on_platform = True
        
        self.particles = []
        
        self._generate_platforms()
        
        start_platform = self.platforms[0]
        self.player_pos = pygame.Vector2(
            start_platform.centerx, start_platform.top - self.player_size
        )
        self.player_rect = pygame.Rect(0, 0, self.player_size, self.player_size)
        self.current_platform_index = 0
        self.last_platform_index = 0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- Handle Input and Game Logic ---
        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._update_particles()

            # Time penalty and reward for staying on platform
            self.timer -= 1
            if self.on_platform:
                reward += 0.1

            # --- Collision Detection ---
            landed_on_new_platform = False
            if self.player_vel.y > 0: # Only check for landing if falling
                self.player_rect.topleft = self.player_pos
                landed_this_frame = False
                for i, plat in enumerate(self.platforms):
                    if self.player_rect.colliderect(plat) and self.player_pos.y + self.player_size - self.player_vel.y <= plat.top:
                        self.player_pos.y = plat.top - self.player_size
                        self.player_vel.y = 0
                        
                        if not self.on_platform: # Fresh landing
                            self._create_landing_particles(pygame.Vector2(self.player_pos.x + self.player_size / 2, plat.top))
                            
                            # Reward for landing on a higher platform
                            if i > self.current_platform_index:
                                reward += 1.0
                            
                            # Penalty for "safe" jump (to platform directly above)
                            old_plat = self.platforms[self.current_platform_index]
                            if abs(plat.centerx - old_plat.centerx) < plat.width / 2:
                                reward -= 0.2

                            # Update platform indices
                            self.last_platform_index = self.current_platform_index
                            self.current_platform_index = i
                            landed_on_new_platform = True

                        self.on_platform = True
                        landed_this_frame = True
                        break
                if not landed_this_frame:
                    self.on_platform = False
            elif self.player_vel.y <= 0 and not self.on_platform:
                # Check if player is still in the air
                self.player_rect.topleft = self.player_pos
                if not any(self.player_rect.colliderect(p) for p in self.platforms):
                    self.on_platform = False
        
        # --- Calculate Score and Termination ---
        self.score += reward
        terminated = self._check_termination()
        
        # Apply terminal rewards
        if terminated:
            if self.current_platform_index == self.top_platform_index and self.on_platform:
                reward = 100 # Win
                self.score += 100
            elif self.player_pos.y > self.SCREEN_HEIGHT:
                reward = -100 # Fall
                self.score -= 100
            elif self.timer <= 0:
                reward = -50 # Time out
                self.score -= 50
        elif landed_on_new_platform and self.current_platform_index == self.top_platform_index:
            reward += 5 # Reached top platform (non-terminal)
            self.score += 5
        
        self.steps += 1
        
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_platforms(self):
        self.platforms = []
        # Start platform
        start_plat = pygame.Rect(self.SCREEN_WIDTH / 2 - 100, self.SCREEN_HEIGHT - 40, 200, 20)
        self.platforms.append(start_plat)

        y = self.SCREEN_HEIGHT - 120
        last_x = start_plat.centerx
        
        while y > 80:
            max_reach = 140
            min_reach = 40
            
            offset = self.np_random.integers(-max_reach, max_reach + 1)
            if abs(offset) < min_reach:
                offset = min_reach * np.sign(offset) if offset != 0 else min_reach

            x = last_x + offset
            width = self.np_random.integers(70, 130)
            
            # Clamp to screen bounds
            x = np.clip(x, width / 2, self.SCREEN_WIDTH - width / 2)
            
            plat = pygame.Rect(x - width / 2, y, width, 15)
            self.platforms.append(plat)
            
            last_x = x
            y -= self.np_random.integers(60, 90)

        # Top platform
        top_plat_width = 150
        top_plat = pygame.Rect(self.SCREEN_WIDTH / 2 - top_plat_width / 2, 50, top_plat_width, 20)
        self.platforms.append(top_plat)
        self.top_platform_index = len(self.platforms) - 1

    def _handle_input(self, action):
        if not self.on_platform:
            return

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        jump_vel = 0
        side_vel = 0

        # Prioritize special jumps (Space > Shift > Movement)
        if space_held: # High jump
            jump_vel = self.JUMP_HIGH
        elif shift_held: # Low jump
            jump_vel = self.JUMP_LOW
        elif movement == 1: # Up (Medium jump)
            jump_vel = self.JUMP_MEDIUM
        elif movement == 3: # Left
            jump_vel = self.JUMP_MEDIUM
            side_vel = -self.JUMP_SIDE_VEL
        elif movement == 4: # Right
            jump_vel = self.JUMP_MEDIUM
            side_vel = self.JUMP_SIDE_VEL
        
        if jump_vel != 0:
            self.player_vel.y = jump_vel
            self.player_vel.x = side_vel
            self.on_platform = False

    def _update_player(self):
        # Apply gravity only when airborne
        if not self.on_platform:
            self.player_vel.y += self.GRAVITY
            self.player_vel.y = min(self.player_vel.y, self.MAX_VEL_Y)

        # Apply horizontal friction/decay
        self.player_vel.x *= 0.9

        # Update position
        self.player_pos += self.player_vel

        # Wall bouncing
        if self.player_pos.x < 0:
            self.player_pos.x = 0
            self.player_vel.x *= -0.5
        if self.player_pos.x > self.SCREEN_WIDTH - self.player_size:
            self.player_pos.x = self.SCREEN_WIDTH - self.player_size
            self.player_vel.x *= -0.5

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['life'] / p['max_life'] * 4)

    def _create_landing_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(0, math.pi * 2)
            speed = self.np_random.uniform(1, 3)
            life = self.np_random.integers(10, 20)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': life,
                'max_life': life,
                'radius': 4
            })
            
    def _check_termination(self):
        if self.game_over:
            return True
        
        # Win/Loss conditions
        if self.current_platform_index == self.top_platform_index and self.on_platform:
            self.game_over = True
            return True
        if self.player_pos.y > self.SCREEN_HEIGHT:
            self.game_over = True
            return True
        if self.timer <= 0:
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw platforms
        for i, plat in enumerate(self.platforms):
            color = self.COLOR_TOP_PLATFORM if i == self.top_platform_index else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, plat, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, plat, 1, border_radius=3)

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), self.COLOR_PARTICLE
            )

        # Draw player
        self.player_rect.topleft = (int(self.player_pos.x), int(self.player_pos.y))
        inner_shrink = 4
        inner_rect = self.player_rect.inflate(-inner_shrink, -inner_shrink)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_INNER, inner_rect, border_radius=2)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_color = (255, 100, 100) if time_left < 5 else self.COLOR_TEXT
        timer_text = self.font_large.render(f"Time: {time_left:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        if self.game_over:
            message = ""
            if self.current_platform_index == self.top_platform_index and self.on_platform:
                message = "YOU WIN!"
            elif self.player_pos.y > self.SCREEN_HEIGHT:
                message = "GAME OVER"
            elif self.timer <= 0:
                message = "TIME'S UP"
            
            if message:
                overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 128))
                self.screen.blit(overlay, (0, 0))

                end_text = self.font_large.render(message, True, self.COLOR_TEXT)
                end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
                self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": round(self.timer / self.FPS, 2),
        }
    
    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play Example ---
    # This part will not run in a typical headless environment
    # but is useful for local testing with a display.
    # To run this, remove the os.environ line in __init__.
    try:
        # Check if we are in headless mode
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            print("Running in headless mode. No interactive window will be shown.")
            print("To play manually, remove 'os.environ[\"SDL_VIDEODRIVER\"] = \"dummy\"' from __init__.")

            # Simple test loop for headless mode
            obs, info = env.reset(seed=42)
            done = False
            total_reward = 0
            step_count = 0
            while not done:
                action = env.action_space.sample() # Random actions
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                done = terminated or truncated
            print(f"Headless test finished in {step_count} steps. Total reward: {total_reward}")
        else:
             # This block is for interactive play and requires a display
            raise RuntimeError("Not in headless mode, interactive play is enabled.")

    except (KeyError, RuntimeError):
        # Interactive mode
        # Re-initialize pygame for display
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("Platformer Game")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        
        obs, info = env.reset(seed=42)
        done = False
        running = True
        
        print(env.user_guide)

        while running:
            if done:
                # Wait for a moment before resetting
                pygame.time.wait(2000)
                obs, info = env.reset(seed=42)
                done = False

            # --- Get human input ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]

            # --- Step the environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- Render to screen ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # --- Event handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            env.clock.tick(env.FPS)
            
        env.close()