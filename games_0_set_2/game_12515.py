import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:00:14.329449
# Source Brief: brief_02515.md
# Brief Index: 2515
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """Represents a single particle in the environment."""
    def __init__(self, pos, ptype, radius, lifespan, speed, split_delay, color_map):
        self.pos = pygame.Vector2(pos)
        self.ptype = ptype
        self.radius = radius
        self.lifespan = lifespan
        self.speed = speed
        self.split_delay = split_delay
        self.color_map = color_map

        self.age = 0
        self.vel = pygame.Vector2(0, 0)
        self.split_timer = self.split_delay
        self.is_splitting = False
        
        self._set_type_properties()

    def _set_type_properties(self):
        """Set properties based on particle type."""
        self.color = self.color_map[self.ptype]
        if self.ptype in ['red', 'blue']:
            self.vel = pygame.Vector2(random.uniform(-0.5, 0.5), self.speed)
        elif self.ptype == 'green':
            self.vel = pygame.Vector2(0, 0)
            self.is_splitting = True
            self.split_timer = self.split_delay

    def transform_to_green(self):
        """Transforms a red particle into a green one."""
        self.ptype = 'green'
        self.age = 0
        self._set_type_properties()
        # Sound: "shimmer.wav"

    def update(self):
        """Update particle state for one frame."""
        self.age += 1
        self.pos += self.vel

        if self.is_splitting:
            self.split_timer -= 1

    def should_split(self):
        return self.is_splitting and self.split_timer <= 0

    def is_dead(self):
        return self.age > self.lifespan

    def draw(self, surface):
        """Draw the particle with a glow effect."""
        x, y = int(self.pos.x), int(self.pos.y)
        
        # Glow effect
        glow_radius = int(self.radius * 1.8)
        glow_alpha = int(100 * (1 - self.age / self.lifespan))
        if glow_alpha > 0:
            glow_color = (*self.color, glow_alpha)
            # Using a simple filled circle for glow as gfxdraw doesn't support alpha well on all surfaces
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            surface.blit(temp_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Core particle
        pygame.gfxdraw.aacircle(surface, x, y, self.radius, self.color)
        pygame.gfxdraw.filled_circle(surface, x, y, self.radius, self.color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Create a chain reaction by colliding blue and red particles to make green ones. "
        "Convert half the particles to green before time runs out."
    )
    user_guide = (
        "Controls: ← to cycle the left stream's color, → to cycle the right stream's color. "
        "Hold Shift to swap the two streams."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 60 * FPS  # 60 seconds
    WIN_PERCENTAGE = 0.5

    # Colors
    COLOR_BG = (10, 15, 20)
    COLOR_RED = (255, 50, 50)
    COLOR_BLUE = (80, 150, 255)
    COLOR_GREEN = (50, 255, 100)
    COLOR_UI_BAR = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 220)

    # Particle settings
    PARTICLE_RADIUS = 6
    PARTICLE_LIFESPAN = 2.5 * FPS
    PARTICLE_SPEED = 1.5
    GREEN_SPLIT_DELAY = 0.5 * FPS
    SPAWN_INTERVAL = 5 # frames
    MAX_PARTICLES = 300

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 32)
        self.font_timer = pygame.font.Font(None, 48)

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.stream_colors = ['red', 'blue']  # [left, right]
        self.stream_spawners = [
            pygame.Vector2(self.SCREEN_WIDTH * 0.25, -self.PARTICLE_RADIUS),
            pygame.Vector2(self.SCREEN_WIDTH * 0.75, -self.PARTICLE_RADIUS)
        ]
        self.spawn_timer = 0
        self.previous_action = np.array([0, 0, 0])
        self.previous_green_percentage_tier = 0
        self.color_map = {'red': self.COLOR_RED, 'blue': self.COLOR_BLUE, 'green': self.COLOR_GREEN}

        # self.reset() # Removed to align with standard gym practice
        # self.validate_implementation() # Removed as it's for dev, not for final env

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.stream_colors = ['red', 'blue']
        self.spawn_timer = 0
        self.previous_action = np.array([0, 0, 0])
        self.previous_green_percentage_tier = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1

        # 1. Handle player input (on rising edge)
        self._handle_input(action)

        # 2. Spawn new particles
        self._spawn_particles()

        # 3. Update all particles (movement, splitting, aging)
        new_blue_particles = self._update_particles()
        self.particles.extend(new_blue_particles)

        # 4. Handle collisions and calculate creation reward
        creation_reward = self._handle_collisions()
        reward += creation_reward

        # 5. Calculate green percentage and reward
        green_percentage = self._get_green_percentage()
        current_tier = int(green_percentage * 100)
        if current_tier > self.previous_green_percentage_tier:
            reward += (current_tier - self.previous_green_percentage_tier) * 1.0
            self.previous_green_percentage_tier = current_tier

        # 6. Check for termination
        terminated = False
        if green_percentage >= self.WIN_PERCENTAGE:
            terminated = True
            reward += 100
            # Sound: "win_fanfare.wav"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 100
            # Sound: "lose_buzzer.wav"
        
        self.game_over = terminated
        self.score += reward

        # Enforce particle limit
        if len(self.particles) > self.MAX_PARTICLES:
            self.particles = self.particles[-self.MAX_PARTICLES:]

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        shift_held = action[2] == 1

        prev_movement = self.previous_action[0]
        prev_shift_held = self.previous_action[2] == 1

        # Cycle left stream on left press
        if movement == 3 and prev_movement != 3:
            self.stream_colors[0] = 'blue' if self.stream_colors[0] == 'red' else 'red'
            # Sound: "blip_left.wav"

        # Cycle right stream on right press
        if movement == 4 and prev_movement != 4:
            self.stream_colors[1] = 'blue' if self.stream_colors[1] == 'red' else 'red'
            # Sound: "blip_right.wav"

        # Swap streams on shift press
        if shift_held and not prev_shift_held:
            self.stream_colors.reverse()
            # Sound: "swap.wav"

        self.previous_action = action

    def _spawn_particles(self):
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self.spawn_timer = self.SPAWN_INTERVAL
            if len(self.particles) < self.MAX_PARTICLES:
                # Left stream
                self.particles.append(self._create_particle(self.stream_spawners[0], self.stream_colors[0]))
                # Right stream
                self.particles.append(self._create_particle(self.stream_spawners[1], self.stream_colors[1]))

    def _create_particle(self, pos, ptype):
        return Particle(pos, ptype, self.PARTICLE_RADIUS, self.PARTICLE_LIFESPAN, 
                        self.PARTICLE_SPEED, self.GREEN_SPLIT_DELAY, self.color_map)

    def _update_particles(self):
        newly_created_blue = []
        particles_to_keep = []
        
        for p in self.particles:
            p.update()
            if p.is_dead() or not (0 < p.pos.x < self.SCREEN_WIDTH and p.pos.y < self.SCREEN_HEIGHT):
                continue # Particle is removed

            if p.should_split():
                # Sound: "split.wav"
                for _ in range(2):
                    new_pos = p.pos + pygame.Vector2(random.uniform(-5, 5), random.uniform(-5, 5))
                    newly_created_blue.append(self._create_particle(new_pos, 'blue'))
                continue # Green particle is removed after splitting

            particles_to_keep.append(p)
        
        self.particles = particles_to_keep
        return newly_created_blue

    def _handle_collisions(self):
        reward = 0
        reds = [p for p in self.particles if p.ptype == 'red']
        blues = [p for p in self.particles if p.ptype == 'blue']
        collision_radius_sq = (self.PARTICLE_RADIUS * 2) ** 2

        for blue_p in blues:
            for red_p in reds:
                if red_p.ptype == 'red': # Check again as it might have been converted
                    if blue_p.pos.distance_squared_to(red_p.pos) < collision_radius_sq:
                        red_p.transform_to_green()
                        reward += 0.01
                        break # One blue particle converts one red per frame
        return reward

    def _get_green_percentage(self):
        total_particles = len(self.particles)
        if total_particles == 0:
            return 0.0
        green_count = sum(1 for p in self.particles if p.ptype == 'green')
        return green_count / total_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Green percentage bar
        bar_width = 300
        bar_height = 20
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = 15
        
        green_percentage = self._get_green_percentage()
        fill_width = int(bar_width * green_percentage)
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_GREEN, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = f"{max(0, time_left):.1f}"
        text_surface = self.font_timer.render(time_text, True, self.COLOR_UI_TEXT)
        text_rect = text_surface.get_rect(midleft=(bar_x + bar_width + 15, bar_y + bar_height / 2))
        self.screen.blit(text_surface, text_rect)

        # Stream color indicators
        for i, spawner in enumerate(self.stream_spawners):
            color = self.color_map[self.stream_colors[i]]
            pygame.gfxdraw.aacircle(self.screen, int(spawner.x), 25, self.PARTICLE_RADIUS + 4, color)
            pygame.gfxdraw.filled_circle(self.screen, int(spawner.x), 25, self.PARTICLE_RADIUS + 4, color)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "green_percentage": self._get_green_percentage(),
            "particle_count": len(self.particles)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Since we are in a headless environment, we need to re-init pygame with a display
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    pygame.quit() # Quit the dummy driver
    pygame.init() # Re-init with a visual driver

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Particle Chain Reaction")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    action = np.array([0, 0, 0]) # [movement, space, shift]

    print("\n--- Manual Control ---")
    print("Left/Right Arrow: Cycle stream color")
    print("Shift: Swap stream colors")
    print("Q or ESC: Quit")
    print("----------------------\n")

    while running:
        # Reset action for this frame
        current_action = np.array([0, 0, 0])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                # The input handling logic in the env relies on rising edges,
                # so we only set the action on keydown.
                if event.key == pygame.K_LEFT:
                    current_action[0] = 3
                if event.key == pygame.K_RIGHT:
                    current_action[0] = 4
        
        # Shift is handled as a held key
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            current_action[2] = 1

        obs, reward, terminated, truncated, info = env.step(current_action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']:.2f}, Green Percentage: {info['green_percentage']:.2f}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()