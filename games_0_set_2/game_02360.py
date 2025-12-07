
# Generated: 2025-08-27T20:07:51.307706
# Source Brief: brief_02360.md
# Brief Index: 2360

        
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
        "Controls: ←→ to move. ↑↓ to select number. Space to shoot."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Solve the math equation by shooting the correct number. You have 3 lives. Clear 20 equations to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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

        # Game constants
        self.MAX_STEPS = 1800 # 60 seconds at 30fps
        self.MAX_MISSES = 3
        self.TARGET_EQUATIONS = 20
        self.PLAYER_SPEED = 8
        self.PROJECTILE_SPEED = 10

        # Colors
        self.COLOR_BG = (21, 30, 39) # Dark blue-gray
        self.COLOR_GRID = (44, 62, 80)
        self.COLOR_PLAYER = (52, 152, 219)
        self.COLOR_PROJECTILE = (236, 240, 241)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_CORRECT = (46, 204, 113)
        self.COLOR_INCORRECT = (231, 76, 60)
        self.COLOR_UI = (189, 195, 199)

        # Fonts
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables are initialized in reset()
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.equations_solved = 0
        self.game_over = False
        
        self.player_x = self.WIDTH // 2
        self.selected_number = 1
        
        self.projectiles = []
        self.particles = []
        
        self.last_space_held = False
        self.last_up_held = False
        self.last_down_held = False
        
        self._generate_equation()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- 1. Handle Input & Update State ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Player movement
        if movement == 3: # Left
            self.player_x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_x += self.PLAYER_SPEED
        self.player_x = np.clip(self.player_x, 20, self.WIDTH - 20)

        # Cycle selected number (on press, not hold)
        up_pressed = movement == 1
        down_pressed = movement == 2
        
        if up_pressed and not self.last_up_held:
            self.selected_number = self.selected_number % 9 + 1
        if down_pressed and not self.last_down_held:
            self.selected_number = (self.selected_number - 2 + 9) % 9 + 1
            
        self.last_up_held = up_pressed
        self.last_down_held = down_pressed
        
        # Shoot projectile (on press, not hold)
        if space_held and not self.last_space_held:
            # sfx: player_shoot.wav
            self.projectiles.append({
                'rect': pygame.Rect(self.player_x - 12, self.HEIGHT - 70, 24, 24),
                'number': self.selected_number,
            })
            self._create_particles(10, (self.player_x, self.HEIGHT - 50), self.COLOR_PROJECTILE, 1, 3, -math.pi/2, 0.5)
        self.last_space_held = space_held

        # --- 2. Update Game Entities ---
        # Update projectiles
        projectiles_to_remove = []
        for proj in self.projectiles:
            proj['rect'].y -= self.PROJECTILE_SPEED
            if proj['rect'].bottom < 0:
                projectiles_to_remove.append(proj)
            
            # Check for collision with the equation
            if proj['rect'].colliderect(self.equation['rect']):
                reward += 0.1 # Small reward for aiming correctly
                if proj['number'] == self.equation['answer']:
                    # Correct Answer
                    # sfx: correct_answer.wav
                    self.score += 10
                    reward += 10
                    self.equations_solved += 1
                    self._create_particles(50, self.equation['rect'].center, self.COLOR_CORRECT, 2, 8, 0, 2 * math.pi)
                else:
                    # Incorrect Answer
                    # sfx: incorrect_answer.wav
                    self.misses += 1
                    reward -= 5
                    self._create_particles(50, self.equation['rect'].center, self.COLOR_INCORRECT, 2, 8, 0, 2 * math.pi)

                projectiles_to_remove.append(proj)
                if self.equations_solved < self.TARGET_EQUATIONS and self.misses < self.MAX_MISSES:
                    self._generate_equation()

        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1

        # --- 3. Check for Termination ---
        self.steps += 1
        terminated = False
        if self.equations_solved >= self.TARGET_EQUATIONS:
            # sfx: win_game.wav
            reward += 50
            terminated = True
        elif self.misses >= self.MAX_MISSES:
            # sfx: lose_game.wav
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        # --- 4. Return Gymnasium 5-tuple ---
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _generate_equation(self):
        answer = self.np_random.integers(2, 10)
        a = self.np_random.integers(1, answer)
        b = answer - a
        text = f"{a} + {b} = ?"
        
        text_surf = self.font_large.render(text, True, self.COLOR_TEXT)
        x = self.np_random.integers(100, self.WIDTH - 100 - text_surf.get_width())
        y = self.np_random.integers(50, self.HEIGHT // 2 - 50)
        
        self.equation = {
            'a': a,
            'b': b,
            'answer': answer,
            'text': text,
            'pos': (x, y),
            'rect': text_surf.get_rect(topleft=(x, y))
        }

    def _get_observation(self):
        # --- Render everything to self.screen ---
        self.screen.fill(self.COLOR_BG)
        
        # Render background grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Render equation
        text_surf = self.font_large.render(self.equation['text'], True, self.COLOR_TEXT)
        self.screen.blit(text_surf, self.equation['pos'])

        # Render projectiles
        for proj in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, proj['rect'].centerx, proj['rect'].centery, 12, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, proj['rect'].centerx, proj['rect'].centery, 12, self.COLOR_PROJECTILE)
            num_surf = self.font_small.render(str(proj['number']), True, self.COLOR_BG)
            self.screen.blit(num_surf, num_surf.get_rect(center=proj['rect'].center))

        # Render player
        player_rect = pygame.Rect(0, 0, 40, 20)
        player_rect.center = (int(self.player_x), self.HEIGHT - 40)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Render selected number indicator
        num_indicator_rect = pygame.Rect(0,0,30,30)
        num_indicator_rect.center = (int(self.player_x), self.HEIGHT - 75)
        pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, num_indicator_rect, border_radius=5)
        num_surf = self.font_medium.render(str(self.selected_number), True, self.COLOR_BG)
        self.screen.blit(num_surf, num_surf.get_rect(center=num_indicator_rect.center))
        
        # Draw cannon
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (self.player_x, self.HEIGHT - 40), (self.player_x, self.HEIGHT-50), 4)

        # Render particles
        for p in self.particles:
            if p['radius'] > 0:
                alpha = int(255 * (p['life'] / p['max_life']))
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

        # Render UI
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score
        score_surf = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))

        # Misses
        miss_text_surf = self.font_medium.render("Lives:", True, self.COLOR_UI)
        self.screen.blit(miss_text_surf, (self.WIDTH - 180, 10))
        for i in range(self.MAX_MISSES):
            color = self.COLOR_INCORRECT if i < self.misses else self.COLOR_GRID
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 80 + i * 25, 25, 8, color)
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 80 + i * 25, 25, 8, color)
            
        # Timer / Steps
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_surf = self.font_medium.render(f"Time: {time_left // 30}", True, self.COLOR_UI)
        self.screen.blit(time_surf, time_surf.get_rect(centerx=self.WIDTH//2, y=10))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.equations_solved >= self.TARGET_EQUATIONS:
                msg = "YOU WIN!"
                color = self.COLOR_CORRECT
            else:
                msg = "GAME OVER"
                color = self.COLOR_INCORRECT
            
            msg_surf = self.font_large.render(msg, True, color)
            overlay.blit(msg_surf, msg_surf.get_rect(center=(self.WIDTH//2, self.HEIGHT//2)))
            self.screen.blit(overlay, (0,0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
            "equations_solved": self.equations_solved,
        }

    def _create_particles(self, count, pos, color, min_speed, max_speed, angle_start, angle_range):
        for _ in range(count):
            angle = angle_start + self.np_random.uniform(0, angle_range)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'radius': self.np_random.uniform(3, 7),
                'color': color,
                'life': life,
                'max_life': life
            })

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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Equation Shooter")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # --- Main Game Loop ---
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame-specific event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Display the observation ---
        # The observation is (H, W, C), but pygame wants (W, H) surface, so we need to transpose
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(30) # Run at 30 FPS

    env.close()