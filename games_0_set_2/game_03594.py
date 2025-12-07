
# Generated: 2025-08-27T23:51:27.379399
# Source Brief: brief_03594.md
# Brief Index: 3594

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker where risk-taking is rewarded and cautious play is penalized."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # --- Visuals ---
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_GRID = (20, 30, 50)
        self.COLOR_PADDLE = (220, 220, 220)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = [
            (224, 64, 81),  # Red
            (73, 190, 170), # Teal
            (89, 110, 219), # Blue
            (247, 201, 74), # Yellow
            (199, 94, 229), # Purple
        ]

        # --- Game Constants ---
        self.PADDLE_WIDTH = 80
        self.PADDLE_HEIGHT = 10
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 6
        self.INITIAL_BALL_SPEED = 4.0
        self.MAX_BALL_SPEED = 8.0
        self.MAX_STEPS = 2000

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.balls_left = 0
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = 0
        self.ball_attached = True
        self.blocks = []
        self.particles = []
        self.reward_popups = []
        self.combo_counter = 0

        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.balls_left = 3
        
        self.paddle_rect = pygame.Rect(
            (self.width - self.PADDLE_WIDTH) // 2, 
            self.height - 30, 
            self.PADDLE_WIDTH, 
            self.PADDLE_HEIGHT
        )
        
        self.ball_attached = True
        self.ball_pos = [self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]
        self.ball_speed = self.INITIAL_BALL_SPEED
        
        self._create_blocks()
        
        self.particles = []
        self.reward_popups = []
        self.combo_counter = 0
        
        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        self.blocks = []
        num_cols = 14
        num_rows = 6
        block_width = 38
        block_height = 15
        total_block_width = num_cols * (block_width + 2)
        start_x = (self.width - total_block_width) // 2
        start_y = 50

        for i in range(num_rows):
            for j in range(num_cols):
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                rect = pygame.Rect(
                    start_x + j * (block_width + 2),
                    start_y + i * (block_height + 2),
                    block_width,
                    block_height,
                )
                self.blocks.append({"rect": rect, "color": color})

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = -0.02  # Time penalty
        
        if not self.game_over:
            self._handle_input(action)
            reward += self._update_game_state()

        self.steps += 1
        
        if len(self.blocks) == 0 and not self.win:
            self.win = True
            self.game_over = True
            reward += 100
        
        if self.balls_left <= 0:
            self.game_over = True

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Paddle movement
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        
        self.paddle_rect.x = max(0, min(self.width - self.PADDLE_WIDTH, self.paddle_rect.x))

        # Ball launch
        if self.ball_attached and space_held:
            self.ball_attached = False
            # Start with a random upward angle
            angle = self.np_random.uniform(math.radians(-120), math.radians(-60))
            self.ball_vel = [math.cos(angle) * self.ball_speed, math.sin(angle) * self.ball_speed]
            # // launch_sound

    def _update_game_state(self):
        step_reward = 0

        # --- Ball Movement ---
        if self.ball_attached:
            self.ball_pos[0] = self.paddle_rect.centerx
            self.ball_pos[1] = self.paddle_rect.top - self.BALL_RADIUS
        else:
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]

        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # --- Wall Collisions ---
        if self.ball_pos[0] <= self.BALL_RADIUS:
            self.ball_pos[0] = self.BALL_RADIUS
            self.ball_vel[0] *= -1
            self.combo_counter = 0
        elif self.ball_pos[0] >= self.width - self.BALL_RADIUS:
            self.ball_pos[0] = self.width - self.BALL_RADIUS
            self.ball_vel[0] *= -1
            self.combo_counter = 0
            
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] *= -1
            self.combo_counter = 0

        # --- Bottom Wall (Lose Ball) ---
        if self.ball_pos[1] >= self.height + self.BALL_RADIUS:
            self.balls_left -= 1
            step_reward -= 10
            self.combo_counter = 0
            self.ball_attached = True
            if self.balls_left > 0:
                # // lose_ball_sound
                self._add_reward_popup("-10", (self.width / 2, self.height / 2))
            else:
                # // game_over_sound
                self._add_reward_popup("GAME OVER", (self.width / 2, self.height / 2))

        # --- Paddle Collision ---
        if not self.ball_attached and self.ball_vel[1] > 0 and ball_rect.colliderect(self.paddle_rect):
            # Ensure ball is above paddle to prevent sticking
            self.ball_pos[1] = self.paddle_rect.top - self.BALL_RADIUS
            
            hit_pos = (self.ball_pos[0] - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            hit_pos = max(-1.0, min(1.0, hit_pos)) # Clamp

            # Reward for paddle hits
            if abs(hit_pos) <= 0.2: # Center 20%
                step_reward += 0.1
                self._add_reward_popup("+0.1", self.ball_pos)
            elif abs(hit_pos) >= 0.6: # Outer 40% (20% on each side)
                step_reward -= 0.2
                self._add_reward_popup("-0.2", self.ball_pos)

            angle = math.radians(90 - 75 * hit_pos) # Map -1 to 165 deg, 0 to 90, 1 to 15
            self.ball_speed = min(self.ball_speed + 0.05, self.MAX_BALL_SPEED)
            self.ball_vel = [-math.cos(angle) * self.ball_speed, -math.sin(angle) * self.ball_speed]
            
            self.combo_counter = 0
            # // paddle_hit_sound

        # --- Block Collisions ---
        for i in range(len(self.blocks) - 1, -1, -1):
            block_data = self.blocks[i]
            if ball_rect.colliderect(block_data["rect"]):
                # // block_break_sound
                self._create_particles(block_data["rect"].center, block_data["color"])
                
                # Determine collision side
                overlap = ball_rect.clip(block_data["rect"])
                if overlap.width < overlap.height:
                    self.ball_vel[0] *= -1
                    self.ball_pos[0] += self.ball_vel[0] # Push out
                else:
                    self.ball_vel[1] *= -1
                    self.ball_pos[1] += self.ball_vel[1] # Push out

                self.score += 10
                step_reward += 1
                self.combo_counter += 1
                
                combo_reward = 0
                if self.combo_counter >= 3:
                    combo_reward = 5
                    step_reward += combo_reward

                self._add_reward_popup(f"+{1 + combo_reward}", block_data["rect"].center)
                
                del self.blocks[i]
                break # Only break one block per frame
        
        # --- Update Particles & Popups ---
        self._update_particles()
        self._update_reward_popups()

        return step_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        for i in range(0, self.width, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.height))
        for i in range(0, self.height, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.width, i))

    def _render_game(self):
        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], p["pos"], int(p["radius"]))

        # Blocks
        for block_data in self.blocks:
            pygame.draw.rect(self.screen, block_data["color"], block_data["rect"])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block_data["color"]), block_data["rect"], 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)
        pygame.draw.rect(self.screen, (255, 255, 255), self.paddle_rect.inflate(-4, -4), border_radius=3)

        # Ball
        if not self.game_over or self.win:
            # Glow effect
            for i in range(4, 0, -1):
                alpha = 80 - i * 20
                color = (*self.COLOR_BALL, alpha)
                temp_surf = pygame.Surface((self.BALL_RADIUS * 2 * i, self.BALL_RADIUS * 2 * i), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (self.BALL_RADIUS * i, self.BALL_RADIUS * i), self.BALL_RADIUS * i)
                self.screen.blit(temp_surf, (int(self.ball_pos[0] - self.BALL_RADIUS*i), int(self.ball_pos[1] - self.BALL_RADIUS*i)))
            
            # Ball itself
            pos = (int(self.ball_pos[0]), int(self.ball_pos[1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:05}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Balls left
        for i in range(self.balls_left):
            pygame.gfxdraw.aacircle(self.screen, self.width - 20 - i*20, 25, 5, self.COLOR_PADDLE)
            pygame.gfxdraw.filled_circle(self.screen, self.width - 20 - i*20, 25, 5, self.COLOR_PADDLE)

        # Reward Popups
        for popup in self.reward_popups:
            alpha = max(0, min(255, int(255 * (popup["life"] / 30))))
            color = (*popup["color"], alpha)
            text_surf = self.font_small.render(str(popup["text"]), True, color)
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=popup["pos"])
            self.screen.blit(text_surf, text_rect)
        
        # Game Messages
        if self.ball_attached and not self.game_over:
            msg = "PRESS SPACE TO LAUNCH"
            text_surf = self.font_small.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.width/2, self.height * 0.75))
            self.screen.blit(text_surf, text_rect)
        
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "radius": self.np_random.uniform(2, 5),
                "life": 20,
                "color": color
            })

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["life"] -= 1
            p["radius"] -= 0.1
            if p["life"] <= 0 or p["radius"] <= 0:
                del self.particles[i]
    
    def _add_reward_popup(self, text, pos):
        color = (0, 255, 0) if isinstance(text, str) and text.startswith('+') or isinstance(text, (int, float)) and text > 0 else (255, 100, 100)
        self.reward_popups.append({"text": text, "pos": list(pos), "life": 30, "color": color})

    def _update_reward_popups(self):
        for i in range(len(self.reward_popups) - 1, -1, -1):
            popup = self.reward_popups[i]
            popup["pos"][1] -= 1
            popup["life"] -= 1
            if popup["life"] <= 0:
                del self.reward_popups[i]
    
    def close(self):
        pygame.font.quit()
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import os
    # Set the display environment variable for different OS
    # For Windows, you might not need to set this.
    # For Linux, it's often "x11".
    # For headless servers, it's "dummy".
    if os.name == 'posix' and "DISPLAY" not in os.environ:
         os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # To play the game manually, a Pygame window is needed.
    try:
        real_screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Block Breaker")
        is_manual_play = True
    except pygame.error:
        print("Pygame display could not be initialized. Running in headless mode.")
        is_manual_play = False

    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        if is_manual_play:
            # Map keyboard for manual control
            keys = pygame.key.get_pressed()
            move_action = 0 # none
            if keys[pygame.K_LEFT]:
                move_action = 3
            elif keys[pygame.K_RIGHT]:
                move_action = 4
            
            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = np.array([move_action, space_action, shift_action])

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset() # Reset on 'r' key
        else:
            # Simple random agent for headless mode
            action = env.action_space.sample()

        if terminated:
            break

        obs, reward, terminated, truncated, info = env.step(action)
        
        if is_manual_play:
            # Update the display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            real_screen.blit(surf, (0, 0))
            pygame.display.flip()
                
    print(f"Game Over! Final Score: {info['score']}")
    env.close()