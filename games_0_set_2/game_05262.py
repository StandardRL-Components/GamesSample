
# Generated: 2025-08-28T04:29:31.191425
# Source Brief: brief_05262.md
# Brief Index: 5262

        
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
        "Controls: Arrow keys to move the reticle. Press Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down target practice game. Hit the moving targets before you run out of shots. Aim carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 30, 48)  # Dark Blue
    COLOR_TARGET = (248, 113, 113)  # Red-400
    COLOR_TARGET_OUTLINE = (220, 38, 38) # Red-600
    COLOR_PROJECTILE = (241, 245, 249)  # Slate-100
    COLOR_RETICLE = (74, 222, 128)  # Green-400
    COLOR_TEXT = (226, 232, 240)  # Slate-200
    COLOR_TEXT_SHADOW = (15, 23, 42) # Slate-900
    
    # Game parameters
    INITIAL_SHOTS = 5
    TARGETS_TO_WIN = 15
    RETICLE_SPEED = 10.0
    PROJECTILE_SPEED = 20.0
    INITIAL_TARGET_SPEED = 1.0
    TARGET_SPEED_INCREASE = 0.05
    TARGET_RADIUS = 15
    PROJECTILE_SIZE = 4
    GUN_POSITION = np.array([WIDTH / 2, HEIGHT])
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.shots_remaining = 0
        self.reticle_pos = np.array([0.0, 0.0])
        self.targets = []
        self.projectiles = []
        self.particles = []
        self.target_base_speed = 0.0
        self.last_space_held = False
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # Uncomment for self-check during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.shots_remaining = self.INITIAL_SHOTS
        self.reticle_pos = np.array([self.WIDTH / 2.0, self.HEIGHT / 2.0])
        self.target_base_speed = self.INITIAL_TARGET_SPEED
        self.last_space_held = False
        
        self.targets = []
        self.projectiles = []
        self.particles = []

        for _ in range(3):
            self._spawn_target()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = False
        self.steps += 1

        if not self.game_over:
            # Unpack factorized action
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            # --- Handle Input ---
            self._handle_input(movement, space_held)

            # --- Update Game Logic ---
            self._update_targets()
            hit_reward, miss_penalty = self._update_projectiles_and_collisions()
            reward += hit_reward + miss_penalty
            self._update_particles()

        # --- Check Termination Conditions ---
        if self.score >= self.TARGETS_TO_WIN:
            terminated = True
            reward += 10.0  # Win bonus
        elif self.shots_remaining <= 0 and not self.projectiles:
            terminated = True
            reward -= 5.0  # Loss penalty
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Timeout

        if terminated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Move reticle
        if movement == 1: self.reticle_pos[1] -= self.RETICLE_SPEED
        elif movement == 2: self.reticle_pos[1] += self.RETICLE_SPEED
        elif movement == 3: self.reticle_pos[0] -= self.RETICLE_SPEED
        elif movement == 4: self.reticle_pos[0] += self.RETICLE_SPEED

        # Clamp reticle to screen
        self.reticle_pos[0] = np.clip(self.reticle_pos[0], 0, self.WIDTH)
        self.reticle_pos[1] = np.clip(self.reticle_pos[1], 0, self.HEIGHT)

        # Fire projectile on key press (not hold)
        if space_held and not self.last_space_held and self.shots_remaining > 0:
            self._fire_projectile()
        self.last_space_held = space_held

    def _fire_projectile(self):
        # sfx: Pew!
        self.shots_remaining -= 1
        
        direction = self.reticle_pos - self.GUN_POSITION
        norm = np.linalg.norm(direction)
        if norm > 0:
            velocity = (direction / norm) * self.PROJECTILE_SPEED
        else: # Reticle is exactly at gun position, fire straight up
            velocity = np.array([0, -self.PROJECTILE_SPEED])
            
        self.projectiles.append({
            "pos": self.GUN_POSITION.copy(),
            "vel": velocity,
        })

    def _update_targets(self):
        for target in self.targets:
            target['angle'] += target['speed'] * target['direction'] / self.FPS
            target['pos'][0] = target['center'][0] + math.cos(target['angle']) * target['radius']
            target['pos'][1] = target['center'][1] + math.sin(target['angle']) * target['radius']

    def _update_projectiles_and_collisions(self):
        hit_reward = 0.0
        miss_penalty = 0.0

        for proj in self.projectiles[:]:
            proj['pos'] += proj['vel']
            
            # Check for collision with targets
            hit = False
            for target in self.targets[:]:
                dist = np.linalg.norm(proj['pos'] - target['pos'])
                if dist < self.TARGET_RADIUS + self.PROJECTILE_SIZE:
                    # sfx: Explosion!
                    hit_reward += 1.0
                    self.score += 1
                    self._spawn_particles(target['pos'], self.COLOR_TARGET, 20)
                    self.targets.remove(target)
                    self.projectiles.remove(proj)
                    hit = True
                    
                    # Increase difficulty every 3 hits
                    if self.score > 0 and self.score % 3 == 0:
                        self.target_base_speed += self.TARGET_SPEED_INCREASE
                    
                    self._spawn_target()
                    break
            if hit:
                continue

            # Check for out of bounds (miss)
            if not (0 <= proj['pos'][0] <= self.WIDTH and 0 <= proj['pos'][1] <= self.HEIGHT):
                miss_penalty -= 0.1
                self.projectiles.remove(proj)
        
        return hit_reward, miss_penalty

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_target(self):
        padding = 50
        radius = self.np_random.uniform(40, 120)
        center_x = self.np_random.uniform(padding + radius, self.WIDTH - padding - radius)
        center_y = self.np_random.uniform(padding + radius, self.HEIGHT - padding - radius)
        
        self.targets.append({
            "pos": np.array([0.0, 0.0]),
            "center": np.array([center_x, center_y]),
            "radius": radius,
            "angle": self.np_random.uniform(0, 2 * math.pi),
            "speed": self.target_base_speed * self.np_random.uniform(0.8, 1.2),
            "direction": self.np_random.choice([-1, 1])
        })

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                "life": self.np_random.integers(10, 20),
                "color": color,
                "size": self.np_random.uniform(2, 5)
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render targets
        for target in self.targets:
            x, y = int(target['pos'][0]), int(target['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.TARGET_RADIUS, self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.TARGET_RADIUS, self.COLOR_TARGET_OUTLINE)

        # Render projectiles
        for proj in self.projectiles:
            x, y = int(proj['pos'][0]), int(proj['pos'][1])
            size = self.PROJECTILE_SIZE
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, (x - size/2, y - size/2, size, size))
            
        # Render particles
        for p in self.particles:
            x, y = int(p['pos'][0]), int(p['pos'][1])
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (x - p['size'], y - p['size']))
            
        # Render reticle
        rx, ry = int(self.reticle_pos[0]), int(self.reticle_pos[1])
        size = 12
        pygame.gfxdraw.hline(self.screen, rx - size, rx + size, ry, self.COLOR_RETICLE)
        pygame.gfxdraw.vline(self.screen, rx, ry - size, ry + size, self.COLOR_RETICLE)
        pygame.gfxdraw.aacircle(self.screen, rx, ry, size // 2, self.COLOR_RETICLE)

    def _render_ui(self):
        def draw_text(text, font, color, pos, shadow=True):
            if shadow:
                text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
                self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Score
        draw_text(f"SCORE: {self.score}", self.font_main, self.COLOR_TEXT, (10, 10))
        
        # Shots
        shots_text = f"SHOTS: {self.shots_remaining}"
        text_width = self.font_main.size(shots_text)[0]
        draw_text(shots_text, self.font_main, self.COLOR_TEXT, (self.WIDTH - text_width - 10, 10))

        # Game Over message
        if self.game_over:
            if self.score >= self.TARGETS_TO_WIN:
                msg = "YOU WIN!"
                color = (134, 239, 172) # Green-300
            elif self.steps >= self.MAX_STEPS:
                msg = "TIME UP"
                color = (253, 186, 116) # Orange-300
            else:
                msg = "GAME OVER"
                color = (252, 165, 165) # Red-300
            
            text_width, text_height = self.font_big.size(msg)
            draw_text(msg, self.font_big, color, (self.WIDTH/2 - text_width/2, self.HEIGHT/2 - text_height/2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "shots_remaining": self.shots_remaining,
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
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
        assert "score" in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert "steps" in info

        # Test state guarantees
        self.reset()
        self.shots_remaining = 1
        self.step(self.action_space.sample()) # Use up the last shot
        assert self.shots_remaining == 0
        # Wait for projectile to disappear
        for _ in range(100):
            obs, reward, term, trunc, info = self.step(self.action_space.sample())
            if term:
                break
        assert term, "Game should terminate after running out of shots"
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Target Practice")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    running = True
    terminated = False
    
    print("\n" + "="*30)
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Human Input ---
        movement = 0 # no-op
        space_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        action = [movement, space_held, 0] # shift is not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

            if terminated:
                print(f"\n--- GAME END ---")
                print(f"Final Score: {info['score']}")
                print(f"Press 'R' to play again.")

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()