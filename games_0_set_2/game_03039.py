
# Generated: 2025-08-28T06:47:09.240144
# Source Brief: brief_03039.md
# Brief Index: 3039

        
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
    """
    A fast-paced arcade game where the player navigates a procedurally generated
    tunnel, dodging obstacles to survive as long as possible.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ↑ and ↓ to move your ship. Avoid the red obstacles."
    )

    game_description = (
        "Navigate a neon tunnel at high speed. Dodge the oncoming red barriers "
        "to score points and survive as long as you can. The further you go, "
        "the faster it gets!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Gymnasium spaces
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
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_WALL_OUTER = (20, 10, 40)
        self.COLOR_WALL_INNER = (40, 20, 80)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (255, 50, 50, 70)
        self.COLOR_PARTICLE = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)

        # Game constants
        self.MAX_STEPS = 1800 # 30 seconds at 60 FPS
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 5
        self.INITIAL_OBSTACLE_SPEED = 3.0
        self.OBSTACLE_SPEED_INCREASE = 0.05
        self.TUNNEL_MARGIN = 40
        
        # State variables will be initialized in reset()
        self.player_rect = None
        self.obstacles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.obstacle_speed = None
        self.last_spawn_x = None
        self.rng = None

        self.reset()

        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = self.np_random

        # Player state
        player_x = self.WIDTH * 0.2
        player_y = self.HEIGHT / 2
        self.player_rect = pygame.Rect(
            player_x - self.PLAYER_SIZE / 2,
            player_y - self.PLAYER_SIZE / 2,
            self.PLAYER_SIZE,
            self.PLAYER_SIZE,
        )

        # Game state
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.last_spawn_x = self.WIDTH + 50
        
        # Pre-spawn initial obstacles
        for i in range(5):
             self._spawn_obstacle_set()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # Unpack action
        movement = action[0]

        # 1. Update Player
        moved = False
        if movement == 1:  # Up
            self.player_rect.y -= self.PLAYER_SPEED
            moved = True
        elif movement == 2:  # Down
            self.player_rect.y += self.PLAYER_SPEED
            moved = True

        # Clamp player position within tunnel walls
        self.player_rect.y = np.clip(
            self.player_rect.y, self.TUNNEL_MARGIN, self.HEIGHT - self.TUNNEL_MARGIN - self.PLAYER_SIZE
        )
        
        # Apply rewards
        reward += 0.01  # Survival reward
        if not moved:
            reward -= 0.02 # Stillness penalty

        # 2. Update Game World
        self._update_obstacles(reward)
        self._update_particles()
        
        # Increase difficulty
        if self.steps > 0 and self.steps % 100 == 0:
            self.obstacle_speed += self.OBSTACLE_SPEED_INCREASE

        # 3. Check for Termination
        # Collision
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs["rect"]):
                self.game_over = True
                reward = -100.0
                # sfx: player_explosion
                self._spawn_particles(self.player_rect.center, self.COLOR_PLAYER, 50, 5)
                break
        
        # Victory / Timeout
        if not self.game_over and self.steps >= self.MAX_STEPS:
            self.game_over = True
            reward = 100.0
            # sfx: victory_fanfare

        # Recalculate total reward from list of rewards
        total_reward = sum(r for r_list in self.obstacles for r in r_list.get("rewards", [])) + reward
        for obs in self.obstacles:
            obs["rewards"] = []


        return (
            self._get_observation(),
            total_reward,
            self.game_over,
            False,
            self._get_info(),
        )

    def _update_obstacles(self, reward):
        player_passed_pos = self.player_rect.left
        
        # Use a copy for safe iteration while removing
        for obs in self.obstacles[:]:
            obs["rect"].x -= self.obstacle_speed
            
            # Remove if off-screen
            if obs["rect"].right < 0:
                self.obstacles.remove(obs)
                self._spawn_obstacle_set()
                continue
            
            # Check for successful dodge
            if not obs["dodged"] and obs["rect"].right < player_passed_pos:
                obs["dodged"] = True
                self.score += 10
                if "rewards" not in obs: obs["rewards"] = []
                obs["rewards"].append(1.0)
                # sfx: dodge_swoosh
                self._spawn_particles(
                    (obs["rect"].right, self.rng.integers(obs["rect"].top, obs["rect"].bottom)),
                    self.COLOR_PARTICLE,
                    10,
                    2
                )
    
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _spawn_obstacle_set(self):
        gap_size = self.rng.integers(100, 150)
        tunnel_height = self.HEIGHT - 2 * self.TUNNEL_MARGIN
        
        gap_y = self.rng.integers(
            self.TUNNEL_MARGIN, self.HEIGHT - self.TUNNEL_MARGIN - gap_size
        )
        
        width = self.rng.integers(20, 40)
        spawn_x = self.last_spawn_x + self.rng.integers(250, 350)
        self.last_spawn_x = spawn_x

        # Top obstacle
        top_height = gap_y - self.TUNNEL_MARGIN
        if top_height > 0:
            self.obstacles.append({
                "rect": pygame.Rect(spawn_x, self.TUNNEL_MARGIN, width, top_height),
                "dodged": False,
                "rewards": []
            })
        
        # Bottom obstacle
        bottom_y = gap_y + gap_size
        bottom_height = self.HEIGHT - self.TUNNEL_MARGIN - bottom_y
        if bottom_height > 0:
            self.obstacles.append({
                "rect": pygame.Rect(spawn_x, bottom_y, width, bottom_height),
                "dodged": False,
                "rewards": []
            })

    def _spawn_particles(self, pos, color, count, power):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, power)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.rng.integers(20, 40),
                "color": color
            })

    def _get_observation(self):
        # 1. Draw background and tunnel
        self.screen.fill(self.COLOR_BG)
        
        # Tunnel perspective walls
        top_pts = [(0, 0), (self.WIDTH, 0), (self.WIDTH, self.TUNNEL_MARGIN), (0, self.TUNNEL_MARGIN)]
        bottom_pts = [(0, self.HEIGHT), (self.WIDTH, self.HEIGHT), (self.WIDTH, self.HEIGHT - self.TUNNEL_MARGIN), (0, self.HEIGHT - self.TUNNEL_MARGIN)]
        
        pygame.gfxdraw.filled_polygon(self.screen, top_pts, self.COLOR_WALL_OUTER)
        pygame.gfxdraw.filled_polygon(self.screen, bottom_pts, self.COLOR_WALL_OUTER)
        
        # 2. Draw particles
        for p in self.particles:
            size = max(1, int(p["life"] * 0.15))
            pygame.draw.rect(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1]), size, size))

        # 3. Draw obstacles
        for obs in self.obstacles:
            # Glow effect
            glow_rect = obs["rect"].inflate(10, 10)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_OBSTACLE_GLOW, s.get_rect(), border_radius=3)
            self.screen.blit(s, glow_rect.topleft)
            # Main obstacle
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs["rect"], border_radius=3)

        # 4. Draw player
        if not self.game_over:
            # Glow effect
            glow_rect = self.player_rect.inflate(self.PLAYER_SIZE * 1.5, self.PLAYER_SIZE * 1.5)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.ellipse(s, self.COLOR_PLAYER_GLOW, s.get_rect())
            self.screen.blit(s, glow_rect.topleft)
            # Main player
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=4)
        
        # 5. Draw UI
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.small_font.render(f"TIME: {self.MAX_STEPS - self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        if self.game_over:
            end_text_str = "LEVEL COMPLETE" if self.steps >= self.MAX_STEPS else "GAME OVER"
            end_text = self.font.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Neon Tunnel")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause before closing
            done = True
            
        clock.tick(60) # Run at 60 FPS
        
    env.close()