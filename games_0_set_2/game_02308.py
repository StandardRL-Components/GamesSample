
# Generated: 2025-08-27T19:58:36.353512
# Source Brief: brief_02308.md
# Brief Index: 2308

        
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
        "Controls: Arrow keys to move. Press space to squash nearby bugs."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Squash waves of procedurally generated bugs before they reach the bottom of the screen in this top-down arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PLAYER_SIZE = 24
        self.PLAYER_SPEED = 8
        self.BUG_SIZE = 12
        self.BUGS_PER_WAVE = 25
        self.INITIAL_LIVES = 5
        self.MAX_STEPS = 10000
        self.SQUASH_RADIUS = 40
        self.SQUASH_DURATION = 4  # frames
        self.SQUASH_COOLDOWN = 12 # frames

        # Colors
        self.COLOR_BG = (30, 100, 40)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_OUTLINE = (255, 255, 255)
        self.COLOR_UI = (255, 255, 200)
        self.BUG_COLORS = {
            "straight": (50, 150, 255),
            "zigzag": (255, 200, 0),
            "sine": (200, 50, 255),
        }

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
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 36)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 20)
        except pygame.error:
            self.font_large = pygame.font.SysFont("monospace", 36)
            self.font_small = pygame.font.SysFont("monospace", 20)

        # Particle system
        self.particles = []
        
        # Initialize state variables
        self.player_pos = None
        self.bugs = None
        self.score = None
        self.lives = None
        self.wave = None
        self.steps = None
        self.game_over = None
        self.squash_timer = None
        self.squash_cooldown_timer = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - self.PLAYER_SIZE * 2], dtype=np.float32)
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.wave = 1
        self.game_over = False
        self.squash_timer = 0
        self.squash_cooldown_timer = 0
        self.particles.clear()
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            # If the game is over, do nothing but return the final state
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = -0.01  # Small penalty for each step to encourage efficiency
        
        self._update_player(movement)
        reward += self._update_squash(space_held, movement != 0)
        reward += self._update_bugs()
        self._update_particles()
        
        self.steps += 1
        
        # Check for wave completion
        if not self.bugs:
            reward += 10
            self.wave += 1
            self._spawn_wave()
            # SFX: Wave clear fanfare
        
        # Check for termination conditions
        terminated = False
        if self.lives <= 0:
            reward -= 10
            self.game_over = True
            terminated = True
            # SFX: Game over sound
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info()
        )

    def _spawn_wave(self):
        self.bugs = []
        bug_speed = 1.0 + (self.wave - 1) * 0.05
        for _ in range(self.BUGS_PER_WAVE):
            bug_type = self.np_random.choice(["straight", "zigzag", "sine"])
            self.bugs.append({
                "pos": np.array([
                    self.np_random.uniform(self.BUG_SIZE, self.WIDTH - self.BUG_SIZE),
                    self.np_random.uniform(-self.HEIGHT, -self.BUG_SIZE)
                ], dtype=np.float32),
                "type": bug_type,
                "speed": bug_speed * self.np_random.uniform(0.8, 1.2),
                "phase": self.np_random.uniform(0, 2 * math.pi),
                "amplitude": self.np_random.uniform(20, 50),
                "frequency": self.np_random.uniform(0.05, 0.1),
                "base_x": 0 # For sine movement
            })
            self.bugs[-1]["base_x"] = self.bugs[-1]["pos"][0]

    def _update_player(self, movement):
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

    def _update_squash(self, space_held, is_moving):
        reward = 0
        if self.squash_cooldown_timer > 0:
            self.squash_cooldown_timer -= 1
        
        if self.squash_timer > 0:
            self.squash_timer -= 1
        
        if space_held and self.squash_cooldown_timer == 0 and self.squash_timer == 0:
            self.squash_timer = self.SQUASH_DURATION
            self.squash_cooldown_timer = self.SQUASH_COOLDOWN
            # SFX: Whoosh sound for squash attempt

        if self.squash_timer > 0:
            # Check for collisions with bugs
            for i in range(len(self.bugs) - 1, -1, -1):
                bug = self.bugs[i]
                dist = np.linalg.norm(self.player_pos - bug["pos"])
                if dist < self.SQUASH_RADIUS:
                    # Squash successful
                    self.score += 10
                    reward += 2.0 if is_moving else -0.2
                    
                    self._create_splat_particles(bug["pos"], self.BUG_COLORS[bug["type"]])
                    
                    del self.bugs[i]
                    # SFX: Bug splat sound
        return reward

    def _update_bugs(self):
        reward = 0
        for i in range(len(self.bugs) - 1, -1, -1):
            bug = self.bugs[i]
            
            # Move bug
            bug["pos"][1] += bug["speed"]
            if bug["type"] == "zigzag":
                bug["pos"][0] += math.sin(bug["phase"] + self.steps * bug["frequency"]) * bug["speed"] * 1.5
            elif bug["type"] == "sine":
                bug["pos"][0] = bug["base_x"] + math.sin(bug["phase"] + bug["pos"][1] * bug["frequency"]) * bug["amplitude"]

            # Bounce off walls
            if not (0 < bug["pos"][0] < self.WIDTH):
                if bug["type"] == "zigzag":
                    bug["frequency"] *= -1 # reverse direction
                bug["pos"][0] = np.clip(bug["pos"][0], 0, self.WIDTH)


            # Check if bug reached the bottom
            if bug["pos"][1] > self.HEIGHT + self.BUG_SIZE:
                self.lives -= 1
                reward -= 1.0
                del self.bugs[i]
                # SFX: Life lost sound
        return reward
        
    def _create_splat_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": velocity,
                "life": self.np_random.integers(10, 20),
                "color": color,
                "size": self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p["pos"] += p["vel"]
            p["vel"] *= 0.95  # friction
            p["life"] -= 1
            if p["life"] <= 0:
                del self.particles[i]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render squash effect
        if self.squash_timer > 0:
            alpha = 100 * (self.squash_timer / self.SQUASH_DURATION)
            radius = self.SQUASH_RADIUS * (1 - self.squash_timer / self.SQUASH_DURATION)
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), int(radius), (*self.COLOR_UI, int(alpha)))
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), int(radius), (*self.COLOR_UI, int(alpha/2)))

        # Render particles
        for p in self.particles:
            alpha = 255 * (p["life"] / 20)
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.draw.circle(self.screen, color, pos, int(p["size"]))

        # Render bugs
        for bug in self.bugs:
            pos = (int(bug["pos"][0]), int(bug["pos"][1]))
            color = self.BUG_COLORS[bug["type"]]
            pygame.draw.circle(self.screen, color, pos, self.BUG_SIZE)
            pygame.draw.circle(self.screen, (0,0,0), pos, self.BUG_SIZE, 1)
            
            # Simple leg animation
            leg_offset = math.sin(self.steps * 0.5 + bug["phase"]) * 4
            pygame.draw.line(self.screen, (0,0,0), (pos[0]-self.BUG_SIZE, pos[1]), (pos[0]-self.BUG_SIZE-leg_offset, pos[1]+5))
            pygame.draw.line(self.screen, (0,0,0), (pos[0]+self.BUG_SIZE, pos[1]), (pos[0]+self.BUG_SIZE+leg_offset, pos[1]+5))

        # Render player
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, 2, border_radius=4)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.wave}", True, self.COLOR_UI)
        wave_rect = wave_text.get_rect(centerx=self.WIDTH/2, top=10)
        self.screen.blit(wave_text, wave_rect)

        # Lives
        for i in range(self.lives):
            pos = (self.WIDTH - 20 - i * (self.BUG_SIZE * 1.5), 20)
            pygame.draw.circle(self.screen, self.BUG_COLORS["straight"], pos, self.BUG_SIZE // 2)
            pygame.draw.circle(self.screen, (0,0,0), pos, self.BUG_SIZE // 2, 1)

        # Game Over message
        if self.game_over:
            game_over_text = self.font_large.render("GAME OVER", True, self.COLOR_PLAYER)
            text_rect = game_over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20,20))
            self.screen.blit(game_over_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "wave": self.wave,
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
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Bug Squasher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    print(env.user_guide)

    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
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
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Info: {info}, Total Reward: {total_reward:.2f}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

    env.close()