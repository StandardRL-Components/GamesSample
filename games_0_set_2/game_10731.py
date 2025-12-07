import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:58:07.806528
# Source Brief: brief_00731.md
# Brief Index: 731
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Bird:
    """Represents a single bird in the flock."""
    def __init__(self, x, y, width, height, np_random):
        self.pos = pygame.Vector2(x, y)
        angle = np_random.uniform(0, 2 * math.pi)
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * np_random.uniform(1.5, 2.5)
        self.acc = pygame.Vector2(0, 0)
        self.max_speed = 4.0
        self.max_force = 0.15
        self.size = 10
        self.screen_width = width
        self.screen_height = height

    def update(self):
        """Updates the bird's position and velocity."""
        self.vel += self.acc
        self.vel.scale_to_length(min(self.vel.length(), self.max_speed))
        self.pos += self.vel
        self.acc *= 0

    def apply_force(self, force):
        """Applies a force to the bird's acceleration."""
        self.acc += force

    def flock(self, birds, player_target_vel):
        """Calculates and applies flocking forces."""
        perception_radius = 60
        separation_radius = 25
        
        separation_force = pygame.Vector2()
        alignment_force = pygame.Vector2()
        cohesion_force = pygame.Vector2()
        
        total_in_perception = 0
        
        for other in birds:
            if other is self:
                continue
            
            dist = self.pos.distance_to(other.pos)
            
            if dist < perception_radius:
                # Cohesion: Steer towards the average position
                cohesion_force += other.pos
                
                # Alignment: Steer towards the average velocity
                alignment_force += other.vel
                
                # Separation: Steer away from close birds
                if dist < separation_radius:
                    diff = self.pos - other.pos
                    if diff.length() > 0:
                        diff.scale_to_length(1 / dist)
                        separation_force += diff
                
                total_in_perception += 1

        if total_in_perception > 0:
            # Finalize Cohesion
            cohesion_force /= total_in_perception
            cohesion_force = cohesion_force - self.pos
            if cohesion_force.length() > 0:
                cohesion_force.scale_to_length(self.max_speed)
            cohesion_force -= self.vel
            cohesion_force.scale_to_length(min(cohesion_force.length(), self.max_force))

            # Finalize Alignment
            alignment_force /= total_in_perception
            if alignment_force.length() > 0:
                alignment_force.scale_to_length(self.max_speed)
            alignment_force -= self.vel
            alignment_force.scale_to_length(min(alignment_force.length(), self.max_force))
            
            # Add player influence to alignment
            alignment_force += player_target_vel * 0.5

            # Finalize Separation
            if separation_force.length() > 0:
                separation_force.scale_to_length(self.max_speed)
            separation_force -= self.vel
            separation_force.scale_to_length(min(separation_force.length(), self.max_force * 1.8)) # Separation is more important

        # Apply weighted forces
        self.apply_force(separation_force * 1.8)
        self.apply_force(alignment_force * 1.2)
        self.apply_force(cohesion_force * 1.0)

    def wrap_edges(self):
        """Wraps the bird around the screen edges."""
        if self.pos.x > self.screen_width:
            self.pos.x = 0
        elif self.pos.x < 0:
            self.pos.x = self.screen_width
        if self.pos.y > self.screen_height:
            self.pos.y = 0
        elif self.pos.y < 0:
            self.pos.y = self.screen_height
            
    def get_bounding_box(self):
        """Gets the approximate bounding box for collision."""
        return pygame.Rect(self.pos.x - self.size/2, self.pos.y - self.size/2, self.size, self.size)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Guide a flock of birds through a field of obstacles. Use your influence to steer the "
        "flock and help as many birds as possible reach the end."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to influence the direction of the flock."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.STARTING_BIRDS = 20
        self.LEVEL_LENGTH = 3000  # Arbitrary units for level progression
        self.MAX_EPISODE_STEPS = 5000
        self.OBSTACLE_SIZE = 50
        
        # Colors
        self.COLOR_BG_TOP = (15, 25, 50)
        self.COLOR_BG_BOTTOM = (30, 50, 100)
        self.COLOR_BIRD = (255, 255, 255)
        self.COLOR_OBSTACLE = (255, 80, 80)
        self.COLOR_OBSTACLE_EDGE = (200, 50, 50)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_PROGRESS_BAR = (100, 120, 200, 150)

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        
        # Persistent state (survives resets until full game over)
        self.level_speed_multiplier = 1.0
        
        # Initialize other state variables
        self.birds = []
        self.obstacles = []
        self.particles = []
        self.player_target_vel = pygame.Vector2(0, 0)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level_progress = 0
        self.obstacle_spawn_timer = 0
        self.current_level_speed = 0
        self.initial_bird_count = 0
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # this is for dev, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Check for full game over (no birds left from previous run)
        if not self.birds:
            self.level_speed_multiplier = 1.0
            self.birds = [
                Bird(
                    self.np_random.uniform(50, 150), 
                    self.np_random.uniform(50, self.HEIGHT - 50), 
                    self.WIDTH, 
                    self.HEIGHT, 
                    self.np_random
                )
                for _ in range(self.STARTING_BIRDS)
            ]
        
        self.initial_bird_count = len(self.birds)
        self.obstacles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level_progress = 0
        self.obstacle_spawn_timer = 0
        self.current_level_speed = 2.0 * self.level_speed_multiplier
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Unpack action and update player intent
        movement = action[0]
        player_force = pygame.Vector2(0, 0)
        if movement == 1: player_force.y = -1  # Up
        elif movement == 2: player_force.y = 1   # Down
        elif movement == 3: player_force.x = -1  # Left
        elif movement == 4: player_force.x = 1   # Right
        
        # Smoothly update target velocity
        self.player_target_vel = self.player_target_vel.lerp(player_force * 0.5, 0.1)

        # 2. Update game logic
        self.steps += 1
        self.level_progress += self.current_level_speed

        self._update_flock()
        self._update_obstacles()
        self._update_particles()
        
        reward = 0.1 # Survival reward
        
        # 3. Handle collisions and update rewards
        crashed_birds, new_particles = self._handle_collisions()
        if crashed_birds:
            self.birds = [b for b in self.birds if b not in crashed_birds]
            self.particles.extend(new_particles)
        
        # 4. Check for termination conditions
        terminated = False
        win = self.level_progress >= self.LEVEL_LENGTH
        lose = not self.birds
        timeout = self.steps >= self.MAX_EPISODE_STEPS

        if win:
            terminated = True
            reward += 1.0  # Base win reward
            if len(self.birds) == self.initial_bird_count:
                reward += 10.0 # Flawless victory
            self.level_speed_multiplier *= 1.05
        elif lose:
            terminated = True
            reward = -100.0 # Severe penalty for losing all birds
        elif timeout:
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        
        truncated = False # No truncation condition in this game
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_flock(self):
        for bird in self.birds:
            bird.flock(self.birds, self.player_target_vel)
            bird.update()
            bird.wrap_edges()
            
    def _update_obstacles(self):
        # Move existing obstacles
        for obstacle in self.obstacles[:]:
            obstacle.x -= self.current_level_speed
            if obstacle.right < 0:
                self.obstacles.remove(obstacle)
        
        # Spawn new obstacles
        self.obstacle_spawn_timer -= 1
        spawn_rate = max(15, 80 / self.level_speed_multiplier)
        if self.obstacle_spawn_timer <= 0:
            y_pos = self.np_random.uniform(0, self.HEIGHT - self.OBSTACLE_SIZE)
            obstacle = pygame.Rect(self.WIDTH, y_pos, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE)
            self.obstacles.append(obstacle)
            self.obstacle_spawn_timer = spawn_rate

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        crashed_birds = set()
        new_particles = []
        for bird in self.birds:
            bird_box = bird.get_bounding_box()
            for obstacle in self.obstacles:
                if bird_box.colliderect(obstacle):
                    crashed_birds.add(bird)
                    # Create a particle burst on collision
                    for _ in range(15):
                        angle = self.np_random.uniform(0, 2 * math.pi)
                        speed = self.np_random.uniform(1, 4)
                        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                        lifespan = self.np_random.integers(20, 41)
                        new_particles.append({'pos': bird.pos.copy(), 'vel': vel, 'lifespan': lifespan})
                    break
        return crashed_birds, new_particles

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "birds_remaining": len(self.birds),
            "level_speed_multiplier": self.level_speed_multiplier,
        }
        
    def _render_all(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_elements(self):
        # Render obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_EDGE, obstacle, 3)

        # Render particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / 40))
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'].x), int(p['pos'].y), 2, (*self.COLOR_BIRD, int(alpha))
            )

        # Render birds
        for bird in self.birds:
            if bird.vel.length() == 0: continue
            
            # Calculate triangle points for the bird
            p1 = bird.pos + bird.vel.normalize() * bird.size
            p2 = bird.pos + bird.vel.normalize().rotate(-140) * bird.size * 0.7
            p3 = bird.pos + bird.vel.normalize().rotate(140) * bird.size * 0.7
            points = [(int(p.x), int(p.y)) for p in (p1, p2, p3)]

            # Draw anti-aliased polygon for a high-quality look
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_BIRD)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_BIRD)

    def _render_ui(self):
        # Render UI Text
        bird_text = self.font_ui.render(f"Birds: {len(self.birds)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(bird_text, (10, 10))
        
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        speed_text = self.font_ui.render(f"Speed: {self.level_speed_multiplier:.2f}x", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (self.WIDTH // 2 - speed_text.get_width() // 2, 10))

        # Render Level Progress Bar
        progress_width = self.WIDTH * (self.level_progress / self.LEVEL_LENGTH)
        progress_rect = pygame.Rect(0, self.HEIGHT - 5, progress_width, 5)
        
        # Use a surface with alpha for a semi-transparent bar
        progress_surface = pygame.Surface((progress_width, 5), pygame.SRCALPHA)
        progress_surface.fill(self.COLOR_PROGRESS_BAR)
        self.screen.blit(progress_surface, (0, self.HEIGHT - 5))

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # It will not work in a headless environment
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False
    
    # Pygame setup for manual play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Flock Environment")
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # [movement, space, shift]

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action[0] = 0 # No movement
        
        # Map keys to actions
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}. Resetting...")
            obs, info = env.reset()
            if done: # If quit was pressed during the final step
                break

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()