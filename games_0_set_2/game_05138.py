
# Generated: 2025-08-28T04:05:06.755843
# Source Brief: brief_05138.md
# Brief Index: 5138

        
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
        "Controls: Use arrow keys (↑↓←→) to move the ninja. Dodge the falling numbers and collect the yellow orbs."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Dodge falling numbers and collect point orbs to become the ultimate Number Ninja! Get a high score by taking risks and collecting orbs near falling numbers."
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
        
        # Colors
        self.COLOR_BG = pygame.Color(15, 15, 25)
        self.COLOR_PLAYER = pygame.Color(255, 255, 255)
        self.COLOR_ORB = pygame.Color(255, 220, 0)
        self.COLOR_NUMBER = pygame.Color(100, 200, 255)
        self.COLOR_DANGER = pygame.Color(255, 80, 80)
        self.COLOR_TEXT = pygame.Color(240, 240, 240)
        self.COLOR_HEART = pygame.Color(220, 40, 40)

        # Fonts
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_number = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("Verdana", 50, bold=True)
        
        # Game constants
        self.PLAYER_SIZE = 10
        self.PLAYER_SPEED = 5
        self.ORB_SIZE = 5
        self.NUMBER_SPEED = 5
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100
        self.STARTING_LIVES = 5
        
        # Initialize state variables to prevent errors before reset
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.last_player_pos = [0, 0]
        self.falling_numbers = []
        self.orbs = []
        self.number_spawn_rate = 0.0
        self.orb_spawn_timer = 0
        
        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.STARTING_LIVES
        self.game_over = False
        
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.last_player_pos = list(self.player_pos)
        
        self.falling_numbers = []
        self.orbs = []
        
        self.number_spawn_rate = 0.01
        self.orb_spawn_timer = 50
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self.last_player_pos = list(self.player_pos)
        
        self._update_player(movement)
        self._update_world()
        
        reward = self._handle_collisions_and_rewards()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
            elif self.lives <= 0:
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # Clamp player position to screen boundaries
        self.player_pos[0] = max(self.PLAYER_SIZE // 2, min(self.WIDTH - self.PLAYER_SIZE // 2, self.player_pos[0]))
        self.player_pos[1] = max(self.PLAYER_SIZE // 2, min(self.HEIGHT - self.PLAYER_SIZE // 2, self.player_pos[1]))

    def _update_world(self):
        # Update and spawn numbers
        self.number_spawn_rate += 0.001
        if self.np_random.random() < self.number_spawn_rate:
            self.falling_numbers.append({
                "pos": [self.np_random.integers(10, self.WIDTH - 10), -10],
                "value": self.np_random.integers(1, 10),
                "size": 20
            })
            # Sound: Number spawn
        
        for num in self.falling_numbers:
            num["pos"][1] += self.NUMBER_SPEED
        
        self.falling_numbers = [n for n in self.falling_numbers if n["pos"][1] < self.HEIGHT + n["size"]]

        # Update and spawn orbs
        self.orb_spawn_timer -= 1
        if self.orb_spawn_timer <= 0:
            self.orbs.append({
                "pos": [self.np_random.integers(20, self.WIDTH - 20), self.np_random.integers(50, self.HEIGHT - 50)],
            })
            self.orb_spawn_timer = 50
            # Sound: Orb spawn

    def _handle_collisions_and_rewards(self):
        reward = 0.0
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_SIZE // 2,
            self.player_pos[1] - self.PLAYER_SIZE // 2,
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        
        # Orb collection
        for orb in self.orbs[:]:
            orb_pos_int = (int(orb['pos'][0]), int(orb['pos'][1]))
            if player_rect.collidepoint(orb_pos_int):
                self.score += 1
                # Sound: Orb collect
                is_risky = any(
                    math.hypot(self.player_pos[0] - num['pos'][0], self.player_pos[1] - num['pos'][1]) < 25
                    for num in self.falling_numbers
                )
                reward += 5.0 if is_risky else 1.0
                self.orbs.remove(orb)

        # Number interaction (collision, proximity, cowardice)
        for num in self.falling_numbers[:]:
            num_rect = pygame.Rect(
                num['pos'][0] - num['size'] // 2,
                num['pos'][1] - num['size'] // 2,
                num['size'], num['size']
            )
            
            if player_rect.colliderect(num_rect):
                self.lives -= 1
                # Sound: Player hit
                self.falling_numbers.remove(num)
                continue

            distance = math.hypot(self.player_pos[0] - num['pos'][0], self.player_pos[1] - num['pos'][1])
            
            if distance < 25:
                reward -= 0.1

            if distance < 50:
                last_distance = math.hypot(self.last_player_pos[0] - num['pos'][0], self.last_player_pos[1] - num['pos'][1])
                if distance > last_distance + 1e-6: # Moved away
                    reward -= 0.20 * max(0, self.score)

        return reward

    def _check_termination(self):
        return self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render orbs
        for orb in self.orbs:
            pulse_radius = self.ORB_SIZE + math.sin(self.steps * 0.2) * 1.5
            pos = (int(orb['pos'][0]), int(orb['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(pulse_radius), self.COLOR_ORB)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(pulse_radius), self.COLOR_ORB)

        # Render numbers
        for num in self.falling_numbers:
            y_ratio = min(1.0, max(0.0, num['pos'][1] / self.HEIGHT))
            color = self.COLOR_NUMBER.lerp(self.COLOR_DANGER, y_ratio)
            pos = (int(num['pos'][0]), int(num['pos'][1]))
            size = int(num['size'])
            
            num_rect = pygame.Rect(pos[0] - size//2, pos[1] - size//2, size, size)
            pygame.draw.rect(self.screen, color, num_rect, border_radius=3)
            
            text_surf = self.font_number.render(str(num['value']), True, self.COLOR_BG)
            text_rect = text_surf.get_rect(center=num_rect.center)
            self.screen.blit(text_surf, text_rect)
            
        # Render player
        player_rect = pygame.Rect(
            self.player_pos[0] - self.PLAYER_SIZE // 2, 
            self.player_pos[1] - self.PLAYER_SIZE // 2, 
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _render_ui(self):
        # Render score
        score_surf = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Render lives as hearts
        for i in range(self.lives):
            heart_pos_x = self.WIDTH - 25 - (i * 25)
            p1 = (heart_pos_x, 22)
            p2 = (heart_pos_x - 10, 12)
            p3 = (heart_pos_x - 5, 12)
            p4 = (heart_pos_x, 17)
            p5 = (heart_pos_x + 5, 12)
            p6 = (heart_pos_x + 10, 12)
            pygame.gfxdraw.filled_polygon(self.screen, [p1,p2,p3,p4,p5,p6], self.COLOR_HEART)
            pygame.gfxdraw.aapolygon(self.screen, [p1,p2,p3,p4,p5,p6], self.COLOR_HEART)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = self.COLOR_ORB if self.score >= self.WIN_SCORE else self.COLOR_DANGER
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
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
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Number Ninja")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # Action mapping for human input
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        # Control frame rate
        clock.tick(30) # 30 FPS

    print(f"Game Over! Final Info: {info}")
    env.close()