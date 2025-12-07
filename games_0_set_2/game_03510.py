
# Generated: 2025-08-27T23:35:09.827465
# Source Brief: brief_03510.md
# Brief Index: 3510

        
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


# Fish class to manage individual fish state and behavior
class Fish:
    def __init__(self, screen_width, screen_height, speed):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Randomize starting side and vertical position
        self.speed = speed * random.choice([-1, 1])
        start_x = -20 if self.speed > 0 else self.screen_width + 20
        start_y = random.randint(50, self.screen_height - 50)
        self.pos = pygame.Vector2(start_x, start_y)
        
        # Sinusoidal movement parameters
        self.amplitude = random.randint(20, 60)
        self.frequency = random.uniform(0.02, 0.05)
        self.initial_y = self.pos.y
        
        self.size = 15
        self.age = 0
        self.color = (255, 165, 0)  # Bright Orange
        self.tail_color = (255, 100, 0)
        
    def update(self):
        self.age += 1
        self.pos.x += self.speed
        # Bobbing motion
        self.pos.y = self.initial_y + self.amplitude * math.sin(self.age * self.frequency)

    def draw(self, surface):
        # Tail flapping animation
        tail_angle = math.sin(self.age * 0.5) * 0.4
        
        # Body
        body_rect = pygame.Rect(self.pos.x - self.size, self.pos.y - self.size/2, self.size * 2, self.size)
        pygame.draw.ellipse(surface, self.color, body_rect)
        
        # Tail
        tail_direction = -1 if self.speed > 0 else 1
        p1 = (self.pos.x - self.size * tail_direction, self.pos.y)
        p2 = (self.pos.x - (self.size + 10) * tail_direction, self.pos.y - 10 * math.cos(tail_angle) - 5)
        p3 = (self.pos.x - (self.size + 10) * tail_direction, self.pos.y + 10 * math.cos(tail_angle) + 5)
        pygame.draw.polygon(surface, self.tail_color, [p1, p2, p3])
        
        # Eye
        eye_pos = (int(self.pos.x + self.size * 0.5 * (-tail_direction)), int(self.pos.y - self.size * 0.2))
        pygame.draw.circle(surface, (0, 0, 0), eye_pos, 2)

    def is_offscreen(self):
        return (self.speed > 0 and self.pos.x > self.screen_width + self.size) or \
               (self.speed < 0 and self.pos.x < -self.size)

    @property
    def rect(self):
        return pygame.Rect(self.pos.x - self.size, self.pos.y - self.size, self.size * 2, self.size * 2)

# Particle class for visual effects
class Particle:
    def __init__(self, x, y, color, life, size_range=(2, 5), speed_range=(1, 3)):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * random.uniform(*speed_range)
        self.color = color
        self.life = life
        self.initial_life = life
        self.size = random.uniform(*size_range)

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95 # Damping
        self.life -= 1
        self.size = max(0, self.size * (self.life / self.initial_life))

    def draw(self, surface):
        if self.life > 0:
            pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), int(self.size))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the net. Catch the fish before they escape!"
    )

    game_description = (
        "Catch 15 fish with a moving net before 3 escape. Faster catches give bonus points."
    )

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
        self.COLOR_BG = (20, 80, 120)
        self.COLOR_BG_WAVE = (15, 60, 100)
        self.COLOR_NET_RIM = (60, 180, 75)
        self.COLOR_NET_MESH = (60, 180, 75, 100) # RGBA
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_MISS = (255, 0, 0)
        self.COLOR_CATCH_PARTICLE = (255, 215, 0)

        # Fonts
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Game state variables are initialized in reset()
        self.net_pos = None
        self.net_speed = 5
        self.net_radius = 30
        self.fishes = []
        self.particles = []
        self.bubbles = []
        self.steps = 0
        self.score = 0
        self.fish_caught = 0
        self.misses = 0
        self.base_fish_speed = 1.0
        self.game_over = False
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.fish_caught = 0
        self.misses = 0
        self.base_fish_speed = 1.0
        self.game_over = False
        
        self.net_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        self.fishes = []
        self.particles = []
        self.bubbles = [self._create_bubble() for _ in range(20)]
        
        # Spawn initial fish
        for _ in range(5):
            self._spawn_fish()
            
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # If auto_advance is True, we tick the clock
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0.0
        
        if self.game_over:
            # If game is over, no actions have effect, just return current state
            terminated = self._check_termination()
            return (
                self._get_observation(),
                0, # No reward after game over
                terminated,
                False,
                self._get_info()
            )

        # 1. Calculate pre-action state for reward shaping
        pre_action_distances = {id(f): self.net_pos.distance_to(f.pos) for f in self.fishes}

        # 2. Process player action
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused
        
        if movement == 1: # Up
            self.net_pos.y -= self.net_speed
        elif movement == 2: # Down
            self.net_pos.y += self.net_speed
        elif movement == 3: # Left
            self.net_pos.x -= self.net_speed
        elif movement == 4: # Right
            self.net_pos.x += self.net_speed
            
        # Clamp net position to screen bounds
        self.net_pos.x = np.clip(self.net_pos.x, self.net_radius, self.WIDTH - self.net_radius)
        self.net_pos.y = np.clip(self.net_pos.y, self.net_radius, self.HEIGHT - self.net_radius)

        # 3. Update game world
        self.steps += 1
        
        # Update fish
        fish_to_remove = []
        for fish in self.fishes:
            fish.update()
            if fish.is_offscreen():
                fish_to_remove.append(fish)
                self.misses += 1
                reward -= 1.0 # Event-based penalty for miss
                # sound: miss_sfx
                self._create_particle_burst(fish.pos.x, fish.pos.y, self.COLOR_MISS, 10)

        # Update particles and bubbles
        self.particles = [p for p in self.particles if p.life > 0]
        self.bubbles = [b for b in self.bubbles if b.life > 0]
        for p in self.particles: p.update()
        for b in self.bubbles: b.update()
        if random.random() < 0.2: # Chance to spawn new bubble
            self.bubbles.append(self._create_bubble())
        
        # 4. Handle collisions (catches)
        caught_fish = []
        for fish in self.fishes:
            if self.net_pos.distance_to(fish.pos) < self.net_radius:
                caught_fish.append(fish)
                self.fish_caught += 1
                self.score += 10
                reward += 1.0 # Event-based reward for catch
                # sound: catch_sfx
                
                # Bonus for fast catch
                if fish.age < 100: # approx 3.3 seconds
                    bonus = 5
                    self.score += bonus
                    reward += 0.5
                
                self._create_particle_burst(fish.pos.x, fish.pos.y, self.COLOR_CATCH_PARTICLE, 30)

                # Increase difficulty
                if self.fish_caught > 0 and self.fish_caught % 5 == 0:
                    self.base_fish_speed += 0.2

        # 5. Clean up and respawn
        self.fishes = [f for f in self.fishes if f not in fish_to_remove and f not in caught_fish]
        for _ in range(len(fish_to_remove) + len(caught_fish)):
            self._spawn_fish()

        # 6. Calculate continuous reward
        post_action_distances = {id(f): self.net_pos.distance_to(f.pos) for f in self.fishes}
        
        # Only compare distances for fish that existed before and after the action
        total_dist_improvement = 0
        for fish_id, old_dist in pre_action_distances.items():
            if fish_id in post_action_distances:
                new_dist = post_action_distances[fish_id]
                total_dist_improvement += (old_dist - new_dist)
        
        # Reward for getting closer, small penalty for moving away
        if total_dist_improvement > 0:
            reward += 0.01 * (total_dist_improvement / self.net_speed) # Normalize by speed
        else:
            reward -= 0.005
        
        # 7. Check for termination
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.fish_caught >= 15:
                self.score += 100
                reward += 10.0 # Goal-oriented reward for winning
            if self.misses >= 3:
                self.score -= 50
                reward -= 10.0 # Goal-oriented penalty for losing
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _spawn_fish(self):
        new_fish = Fish(self.WIDTH, self.HEIGHT, self.base_fish_speed)
        self.fishes.append(new_fish)

    def _create_particle_burst(self, x, y, color, count):
        for _ in range(count):
            self.particles.append(Particle(x, y, color, life=random.randint(20, 40)))

    def _create_bubble(self):
        p = Particle(
            random.randint(0, self.WIDTH),
            random.randint(self.HEIGHT, self.HEIGHT + 100),
            (255, 255, 255, 30), # Semi-transparent white
            life=random.randint(100, 300),
            size_range=(1, 8),
            speed_range=(0.2, 0.8)
        )
        p.vel = pygame.Vector2(random.uniform(-0.1, 0.1), -1) # Mostly upwards
        return p

    def _check_termination(self):
        return self.fish_caught >= 15 or self.misses >= 3 or self.steps >= 1000

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fish_caught": self.fish_caught,
            "misses": self.misses,
        }

    def _render_game(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_BG_WAVE, (0, self.HEIGHT - 50, self.WIDTH, 50))

        # Bubbles (drawn first, behind everything)
        for bubble in self.bubbles:
            # Use a temporary surface for alpha drawing
            s = pygame.Surface((int(bubble.size)*2, int(bubble.size)*2), pygame.SRCALPHA)
            pygame.draw.circle(s, bubble.color, (int(bubble.size), int(bubble.size)), int(bubble.size))
            self.screen.blit(s, (bubble.pos.x - bubble.size, bubble.pos.y - bubble.size))

        # Fishes
        for fish in self.fishes:
            fish.draw(self.screen)
            
        # Particles
        for particle in self.particles:
            particle.draw(self.screen)

        # Net
        net_pos_int = (int(self.net_pos.x), int(self.net_pos.y))
        # Use a temporary surface for transparent net mesh
        s = pygame.Surface((self.net_radius*2, self.net_radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, self.net_radius, self.net_radius, self.net_radius, self.COLOR_NET_MESH)
        self.screen.blit(s, (net_pos_int[0] - self.net_radius, net_pos_int[1] - self.net_radius))
        pygame.gfxdraw.aacircle(self.screen, net_pos_int[0], net_pos_int[1], self.net_radius, self.COLOR_NET_RIM)
        pygame.gfxdraw.aacircle(self.screen, net_pos_int[0], net_pos_int[1], self.net_radius - 1, self.COLOR_NET_RIM)

    def _render_ui(self):
        # Fish Caught UI
        fish_text = self.font_large.render(f"Caught: {self.fish_caught}/15", True, self.COLOR_UI_TEXT)
        self.screen.blit(fish_text, (10, 10))
        
        # Misses UI
        miss_text = self.font_large.render(f"Misses: {self.misses}/3", True, self.COLOR_UI_TEXT)
        miss_rect = miss_text.get_rect(topright=(self.WIDTH - 10, 10))
        # Change color to red if at 2 misses
        if self.misses >= 2:
            miss_text = self.font_large.render(f"Misses: {self.misses}/3", True, self.COLOR_MISS)
        self.screen.blit(miss_text, miss_rect)
        
        # Score UI
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(midtop=(self.WIDTH / 2, 10))
        self.screen.blit(score_text, score_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            if self.fish_caught >= 15:
                end_text_str = "YOU WIN!"
                color = self.COLOR_CATCH_PARTICLE
            else:
                end_text_str = "GAME OVER"
                color = self.COLOR_MISS
            
            end_text = self.font_large.render(end_text_str, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(end_text, end_rect)

            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)


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

# Example of how to run the environment for human play
if __name__ == '__main__':
    import time
    
    env = GameEnv(render_mode="rgb_array")
    env.auto_advance = True 
    
    obs, info = env.reset()
    
    # Setup a window to display the environment
    pygame.display.set_caption("Fish Catcher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    running = True
    
    # Use a no-op action as default
    action = np.array([0, 0, 0])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        
        # Reset action
        action[0] = 0 # No movement
        
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Space and Shift are not used in this game
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # If game is over, allow reset with 'R' key
        if terminated and keys[pygame.K_r]:
            obs, info = env.reset()
            terminated = False

        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()