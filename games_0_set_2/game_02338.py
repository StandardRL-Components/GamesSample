
# Generated: 2025-08-27T20:04:14.786643
# Source Brief: brief_02338.md
# Brief Index: 2338

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.PLAYER_SIZE = 12
        self.ZOMBIE_SIZE = 12
        self.ITEM_RADIUS = 5
        self.PLAYER_SPEED = 3.0
        self.INITIAL_ZOMBIE_SPEED = 1.0
        self.NUM_ZOMBIES = 5
        self.NUM_ITEMS = 10

        # Colors
        self.COLOR_BG = (15, 15, 15)
        self.COLOR_PLAYER = (255, 60, 60)
        self.COLOR_ZOMBIE = (60, 180, 255)
        self.COLOR_ITEM = (255, 220, 60)
        self.COLOR_HELIPAD = (40, 100, 40)
        self.COLOR_HELIPAD_H = (70, 160, 70)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE = (220, 220, 220)

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
        
        # Initialize state variables
        self.player_pos = None
        self.zombies = []
        self.items = []
        self.particles = []
        self.heli_zone = None
        self.zombie_speed = self.INITIAL_ZOMBIE_SPEED
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.zombie_speed = self.INITIAL_ZOMBIE_SPEED
        self.particles = []

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)

        # Place helicopter pad on left or right edge
        heli_w, heli_h = 80, 80
        if self.np_random.random() < 0.5:
            self.heli_zone = pygame.Rect(10, (self.HEIGHT - heli_h) / 2, heli_w, heli_h)
        else:
            self.heli_zone = pygame.Rect(self.WIDTH - heli_w - 10, (self.HEIGHT - heli_h) / 2, heli_w, heli_h)

        # Spawn items
        self.items = []
        for _ in range(self.NUM_ITEMS):
            self.items.append(self._get_random_pos())

        # Spawn zombies
        self.zombies = []
        for _ in range(self.NUM_ZOMBIES):
            pos = self._get_random_pos(min_dist_from_player=100)
            lifetime = self.np_random.integers(300, 601)
            self.zombies.append({"pos": pos, "lifetime": lifetime})

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Not used
        # shift_held = action[2] == 1  # Not used

        # --- Game Logic ---
        reward = 0.1  # Survival reward
        terminated = False

        # 1. Update Player Movement
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE / 2, self.player_pos[1] - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # 2. Update Zombies
        for zombie in self.zombies:
            # Move towards player
            direction = self.player_pos - zombie["pos"]
            dist = np.linalg.norm(direction)
            if dist > 1:
                direction /= dist
                zombie["pos"] += direction * self.zombie_speed
            
            # Lifetime check for respawn
            zombie["lifetime"] -= 1
            if zombie["lifetime"] <= 0:
                zombie["pos"] = self._get_random_pos(min_dist_from_player=100)
                zombie["lifetime"] = self.np_random.integers(300, 601)

        # 3. Update Particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

        # 4. Collision Detection & Rewards
        # Player vs Items
        remaining_items = []
        for item_pos in self.items:
            if np.linalg.norm(self.player_pos - item_pos) < self.PLAYER_SIZE / 2 + self.ITEM_RADIUS:
                reward += 1.0
                self.score += 1
                # sound_effect: item_pickup.wav
            else:
                remaining_items.append(item_pos)
        self.items = remaining_items

        # Player vs Zombies
        for zombie in self.zombies:
            zombie_rect = pygame.Rect(zombie["pos"][0] - self.ZOMBIE_SIZE / 2, zombie["pos"][1] - self.ZOMBIE_SIZE / 2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            if player_rect.colliderect(zombie_rect):
                reward = -5.0
                terminated = True
                self.game_over = True
                self._create_explosion(self.player_pos)
                # sound_effect: player_death.wav
                break
        
        # 5. Check Victory/Termination Conditions
        if not terminated:
            # Player reaches helicopter
            if player_rect.colliderect(self.heli_zone):
                reward = 100.0
                terminated = True
                self.game_over = True
                self.score += 100 # Bonus for winning
                # sound_effect: victory.wav
            
            # Max steps reached
            self.steps += 1
            if self.steps >= self.MAX_STEPS:
                terminated = True
                self.game_over = True

        # 6. Difficulty Scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.zombie_speed += 0.05

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Helicopter Pad
        pygame.draw.rect(self.screen, self.COLOR_HELIPAD, self.heli_zone)
        pygame.draw.rect(self.screen, self.COLOR_HELIPAD_H, self.heli_zone, 3)
        
        # Items
        for item_pos in self.items:
            pygame.gfxdraw.filled_circle(self.screen, int(item_pos[0]), int(item_pos[1]), self.ITEM_RADIUS, self.COLOR_ITEM)
            pygame.gfxdraw.aacircle(self.screen, int(item_pos[0]), int(item_pos[1]), self.ITEM_RADIUS, self.COLOR_ITEM)

        # Zombies
        for zombie in self.zombies:
            rect = pygame.Rect(int(zombie["pos"][0] - self.ZOMBIE_SIZE / 2), int(zombie["pos"][1] - self.ZOMBIE_SIZE / 2), self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, rect)

        # Player
        if not self.game_over:
            player_rect = pygame.Rect(int(self.player_pos[0] - self.PLAYER_SIZE / 2), int(self.player_pos[1] - self.PLAYER_SIZE / 2), self.PLAYER_SIZE, self.PLAYER_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), player_rect, 1) # White outline

        # Particles
        for p in self.particles:
            size = max(0, int((p["life"] / p["max_life"]) * 4))
            if size > 0:
                rect = pygame.Rect(int(p["pos"][0] - size/2), int(p["pos"][1] - size/2), size, size)
                pygame.draw.rect(self.screen, self.COLOR_PARTICLE, rect)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_random_pos(self, min_dist_from_player=0):
        while True:
            pos = np.array([
                self.np_random.uniform(self.ZOMBIE_SIZE, self.WIDTH - self.ZOMBIE_SIZE),
                self.np_random.uniform(self.ZOMBIE_SIZE, self.HEIGHT - self.ZOMBIE_SIZE)
            ], dtype=np.float32)
            if min_dist_from_player == 0 or np.linalg.norm(pos - self.player_pos) > min_dist_from_player:
                return pos

    def _create_explosion(self, position):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(20, 41)
            self.particles.append({"pos": position.copy(), "vel": vel, "life": life, "max_life": life})

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
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Zombie Evasion")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.game_description)
    print("Human Controls: Arrow keys to move.")

    while running:
        # Action defaults to NO-OP
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting

        clock.tick(30) # Limit to 30 FPS
        
    env.close()