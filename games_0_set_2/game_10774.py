import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:01:36.282616
# Source Brief: brief_00774.md
# Brief Index: 774
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for Crates
class Crate:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.radius = 12
        
        crate_type = random.choice(['red', 'green', 'blue'])
        if crate_type == 'red':
            self.type = 'red'
            self.value = 1
            self.color = (255, 80, 80)
            self.glow_color = (255, 80, 80, 50)
        elif crate_type == 'green':
            self.type = 'green'
            self.value = 2
            self.color = (80, 255, 80)
            self.glow_color = (80, 255, 80, 50)
        else: # blue
            self.type = 'blue'
            self.value = 3
            self.color = (80, 150, 255)
            self.glow_color = (80, 150, 255, 50)

        self.pos = [
            random.uniform(self.radius, self.screen_width - self.radius),
            random.uniform(self.radius, self.screen_height - self.radius)
        ]
        self.vel = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        self.bob_angle = random.uniform(0, 2 * math.pi)

    def update(self):
        # Drifting movement
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        # Bounce off walls
        if self.pos[0] <= self.radius or self.pos[0] >= self.screen_width - self.radius:
            self.vel[0] *= -1
        if self.pos[1] <= self.radius or self.pos[1] >= self.screen_height - self.radius:
            self.vel[1] *= -1
        
        # Clamp position to be safe
        self.pos[0] = np.clip(self.pos[0], self.radius, self.screen_width - self.radius)
        self.pos[1] = np.clip(self.pos[1], self.radius, self.screen_height - self.radius)
        
        # Bobbing effect
        self.bob_angle += 0.05

    def draw(self, surface):
        x, y = int(self.pos[0]), int(self.pos[1] + math.sin(self.bob_angle) * 2)
        
        # Glow effect
        glow_radius = int(self.radius * 1.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.glow_color, (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Main crate body
        pygame.gfxdraw.filled_circle(surface, x, y, self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, x, y, self.radius, (255, 255, 255))
        
        # Inner highlight
        pygame.gfxdraw.filled_circle(surface, x-3, y-3, 3, (255, 255, 255, 100))


# Helper class for Particles
class Particle:
    def __init__(self, pos, color):
        self.pos = list(pos)
        self.vel = [random.uniform(-2, 2), random.uniform(-2, 2)]
        self.color = color
        self.lifespan = random.uniform(20, 40)
        self.radius = self.lifespan / 4

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[0] *= 0.95
        self.vel[1] *= 0.95
        self.lifespan -= 1
        self.radius = max(0, self.lifespan / 4)

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / 40))
            color = (*self.color, alpha)
            temp_surf = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (int(self.pos[0] - self.radius), int(self.pos[1] - self.radius)), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Pilot your ship to collect floating crates. Gather sets of red, green, and blue "
        "crates for a combo bonus, but watch your fuel!"
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to move your ship and collect the crates."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)

        # Game constants
        self.MAX_STEPS = 2000
        self.WIN_SCORE = 200
        self.MAX_FUEL = 1000
        self.FUEL_PER_CRATE = 100 # This is a cost, not a gain
        self.INITIAL_CRATES = 10
        self.COMBO_REWARD = 5.0
        self.WIN_REWARD = 100.0
        self.LOSE_REWARD = -100.0
        self.CRATE_COLLECTION_REWARD = 0.1

        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 255, 255, 40)
        self.COLOR_STAR = (200, 200, 220)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_FUEL_BORDER = (100, 100, 120)
        self.COLOR_FUEL_GREEN = (0, 255, 0)
        self.COLOR_FUEL_YELLOW = (255, 255, 0)
        self.COLOR_FUEL_RED = (255, 0, 0)
        
        # Initialize state variables
        self.player_pos = None
        self.player_radius = 15
        self.player_speed = 5
        self.fuel = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.crates = []
        self.particles = []
        self.stars = []
        self.combo_counts = {}

        # self.reset() is called by the wrapper/runner
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fuel = self.MAX_FUEL
        self.player_pos = [self.width / 2, self.height / 2]
        self.combo_counts = {'red': 0, 'green': 0, 'blue': 0}

        # Clear lists
        self.crates.clear()
        self.particles.clear()
        self.stars.clear()

        # Create starfield
        for _ in range(150):
            self.stars.append([
                random.randint(0, self.width),
                random.randint(0, self.height),
                random.choice([1, 2, 2, 3]) # size/speed
            ])

        # Spawn initial crates
        for _ in range(self.INITIAL_CRATES):
            self._spawn_crate(is_initial_spawn=True)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        
        # Update game logic
        self.steps += 1
        reward = 0
        terminated = False

        self._handle_player_movement(movement)
        self._update_game_elements()
        
        collection_reward, combo_reward = self._handle_collisions()
        reward += collection_reward + combo_reward
        
        # Check termination conditions
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += self.WIN_REWARD
        elif self.fuel <= 0:
            terminated = True
            reward += self.LOSE_REWARD
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Truncated is always False in this env
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        if movement == 1: # Up
            self.player_pos[1] -= self.player_speed
        elif movement == 2: # Down
            self.player_pos[1] += self.player_speed
        elif movement == 3: # Left
            self.player_pos[0] -= self.player_speed
        elif movement == 4: # Right
            self.player_pos[0] += self.player_speed
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_radius, self.width - self.player_radius)
        self.player_pos[1] = np.clip(self.player_pos[1], self.player_radius, self.height - self.player_radius)

    def _update_game_elements(self):
        # Update stars
        for star in self.stars:
            star[1] = (star[1] + star[2] * 0.2) % self.height

        # Update crates
        for crate in self.crates:
            crate.update()

        # Update particles
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _handle_collisions(self):
        collection_reward = 0
        combo_reward = 0
        
        for crate in self.crates[:]:
            dist = math.hypot(self.player_pos[0] - crate.pos[0], self.player_pos[1] - crate.pos[1])
            if dist < self.player_radius + crate.radius:
                self.score += crate.value
                self.fuel = max(0, self.fuel - self.FUEL_PER_CRATE)
                self.combo_counts[crate.type] += 1
                collection_reward += self.CRATE_COLLECTION_REWARD

                # Check for combo
                if all(v >= 1 for v in self.combo_counts.values()):
                    self.score += 10 # Bonus points for combo
                    combo_reward += self.COMBO_REWARD
                    self.combo_counts = {'red': 0, 'green': 0, 'blue': 0}

                # Create particle explosion
                for _ in range(20):
                    self.particles.append(Particle(crate.pos, crate.color))
                
                self.crates.remove(crate)
                self._spawn_crate()
        
        return collection_reward, combo_reward

    def _spawn_crate(self, is_initial_spawn=False):
        new_crate = Crate(self.width, self.height)
        
        # Ensure it doesn't spawn on top of the player
        if not is_initial_spawn:
            while math.hypot(self.player_pos[0] - new_crate.pos[0], self.player_pos[1] - new_crate.pos[1]) < 100:
                new_crate.pos = [
                    random.uniform(new_crate.radius, self.width - new_crate.radius),
                    random.uniform(new_crate.radius, self.height - new_crate.radius)
                ]
        self.crates.append(new_crate)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
            "combo": self.combo_counts.copy()
        }

    def _render_game(self):
        # Render stars
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (int(x), int(y), size, size))

        # Render crates
        for crate in self.crates:
            crate.draw(self.screen)
        
        # Render particles
        for p in self.particles:
            p.draw(self.screen)

        # Render player
        player_x, player_y = int(self.player_pos[0]), int(self.player_pos[1])
        
        # Player glow
        glow_radius = int(self.player_radius * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_x - glow_radius, player_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Player body
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, self.player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, self.player_radius, (255, 255, 255))

    def _render_ui(self):
        # Render score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render fuel gauge
        fuel_text = self.font.render("FUEL", True, self.COLOR_UI_TEXT)
        self.screen.blit(fuel_text, (self.width - 180, 10))
        
        fuel_bar_width = 100
        fuel_bar_height = 15
        fuel_bar_x = self.width - fuel_bar_width - 10
        fuel_bar_y = 10 + (self.font.get_height() - fuel_bar_height) // 2

        fuel_ratio = self.fuel / self.MAX_FUEL
        
        # Determine fuel color
        if fuel_ratio > 0.5:
            fuel_color = self.COLOR_FUEL_GREEN
        elif fuel_ratio > 0.2:
            fuel_color = self.COLOR_FUEL_YELLOW
        else:
            fuel_color = self.COLOR_FUEL_RED
        
        # Draw fuel bar
        pygame.draw.rect(self.screen, self.COLOR_FUEL_BORDER, (fuel_bar_x - 2, fuel_bar_y - 2, fuel_bar_width + 4, fuel_bar_height + 4), 2)
        if fuel_ratio > 0:
            pygame.draw.rect(self.screen, fuel_color, (fuel_bar_x, fuel_bar_y, int(fuel_bar_width * fuel_ratio), fuel_bar_height))

        # Render combo tracker
        combo_y = 40
        for i, (ctype, count) in enumerate(self.combo_counts.items()):
            color = {'red': (255, 80, 80), 'green': (80, 255, 80), 'blue': (80, 150, 255)}[ctype]
            is_collected = count > 0
            
            box_x = 10 + i * 25
            pygame.draw.rect(self.screen, color if is_collected else (50,50,70), (box_x, combo_y, 20, 20))
            pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (box_x, combo_y, 20, 20), 1)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Crate Collector")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        movement_action = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement_action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement_action = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement_action = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement_action = 4

        # These actions are part of the action space but not used in game logic
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(60) # Run at 60 FPS for smooth manual play

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()