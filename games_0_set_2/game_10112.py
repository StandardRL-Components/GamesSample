import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:49:46.451838
# Source Brief: brief_00112.md
# Brief Index: 112
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A rhythmic platformer where you time your jumps to land on disappearing platforms. "
        "Select your jump height and leap at the right moment to score."
    )
    user_guide = (
        "Use ↑/↓ arrows to select jump power. Press space to jump. "
        "Time your landings on the green platforms to score."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30 # For physics simulation rate

    # Colors
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (40, 0, 60)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 50)
    COLOR_PLATFORM = (0, 255, 150)
    COLOR_PLATFORM_GLOW = (0, 255, 150, 40)
    COLOR_TEXT = (255, 255, 100)
    COLOR_MISS = (255, 50, 50)
    
    # Player Physics
    GRAVITY = 0.8
    PLAYER_X = WIDTH // 4
    GROUND_Y = HEIGHT - 60
    PLAYER_SIZE = (20, 30)
    JUMP_STRENGTHS = [-14, -17, -20] # Low, Medium, High
    
    # Platform Mechanics
    PLATFORM_COUNT = 3
    PLATFORM_SIZE = (100, 12)
    PLATFORM_CYCLE_FRAMES = 75
    PLATFORM_VISIBLE_DURATION = 45
    PLATFORM_CYCLE_OFFSET = PLATFORM_CYCLE_FRAMES // PLATFORM_COUNT
    PLATFORM_Y_POSITIONS = [GROUND_Y - 60, GROUND_Y - 130, GROUND_Y - 200]
    
    # Game Rules
    MAX_MISSES = 3
    WIN_SCORE = 80
    MAX_STEPS = 1500

    class Particle:
        """A simple particle class for visual effects."""
        def __init__(self, pos, vel, radius, color, lifespan, gravity=0.1):
            self.pos = pygame.Vector2(pos)
            self.vel = pygame.Vector2(vel)
            self.radius = radius
            self.color = color
            self.lifespan = lifespan
            self.max_lifespan = lifespan
            self.gravity = gravity

        def update(self):
            self.pos += self.vel
            self.vel.y += self.gravity
            self.lifespan -= 1
            # Fade out alpha and radius
            self.radius = max(0, self.radius - 0.2)
            
        def draw(self, surface):
            if self.lifespan > 0:
                alpha = int(255 * (self.lifespan / self.max_lifespan))
                # Create a temporary surface for alpha blending
                s = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.color, alpha), (self.radius, self.radius), self.radius)
                surface.blit(s, (int(self.pos.x - self.radius), int(self.pos.y - self.radius)))


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.is_jumping = None
        self.selected_jump_index = None
        self.platforms = None
        self.particles = None
        self.steps = None
        self.score = None
        self.misses = None
        self.game_over = None
        self.prev_space_held = None
        self.prev_movement_action = None
        
        # self.reset() # reset() is called by the wrapper/runner
        # self.validate_implementation() # Validation is for dev, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.PLAYER_X, self.GROUND_Y)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_jumping = False
        self.selected_jump_index = 0
        
        self.platforms = []
        for i in range(self.PLATFORM_COUNT):
            self.platforms.append({
                "rect": pygame.Rect(
                    self.PLAYER_X - self.PLATFORM_SIZE[0] / 2,
                    self.PLATFORM_Y_POSITIONS[i],
                    self.PLATFORM_SIZE[0],
                    self.PLATFORM_SIZE[1]
                ),
                "is_visible": False,
                "cycle_offset": i * self.PLATFORM_CYCLE_OFFSET,
                "points": (i * 2) + 1 # 1, 3, 5 points
            })
            
        self.particles = []
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_movement_action = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        self._handle_input(action)
        self._update_player_state()
        landing_reward, miss_event = self._handle_collisions()
        reward += landing_reward
        
        self._update_platforms()
        self._update_particles()
        
        self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win reward
            elif self.misses >= self.MAX_MISSES:
                reward += -100 # Loss reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Debounce movement action to avoid rapid switching
        if movement != self.prev_movement_action:
            if not self.is_jumping:
                if movement == 1: # Up
                    self.selected_jump_index = min(len(self.JUMP_STRENGTHS) - 1, self.selected_jump_index + 1)
                elif movement == 2: # Down
                    self.selected_jump_index = max(0, self.selected_jump_index - 1)
        self.prev_movement_action = movement

        # Jump on rising edge of space press
        if space_held and not self.prev_space_held and not self.is_jumping:
            self.is_jumping = True
            self.player_vel.y = self.JUMP_STRENGTHS[self.selected_jump_index]
            # SFX: Jump sound based on selected_jump_index
            self._create_particles(self.player_pos + pygame.Vector2(0, self.PLAYER_SIZE[1]/2), 10, self.COLOR_PLAYER, 'jump')

        self.prev_space_held = space_held

    def _update_player_state(self):
        if self.is_jumping:
            self.player_vel.y += self.GRAVITY
            self.player_pos += self.player_vel

    def _handle_collisions(self):
        reward = 0
        missed = False
        
        # Check for landing
        if self.is_jumping and self.player_vel.y > 0 and self.player_pos.y >= self.GROUND_Y:
            landed_on_platform = False
            player_bottom_rect = pygame.Rect(self.player_pos.x - 5, self.player_pos.y + self.PLAYER_SIZE[1]/2 - 5, 10, 10)

            for plat in self.platforms:
                if plat["is_visible"] and plat["rect"].colliderect(player_bottom_rect):
                    # Successful landing
                    self.score += plat["points"]
                    reward += plat["points"] + 0.1 # Event reward + continuous feedback
                    landed_on_platform = True
                    # SFX: Success ping
                    self._create_particles(plat["rect"].midbottom, 20, self.COLOR_PLATFORM, 'land')
                    break
            
            if not landed_on_platform:
                # Missed
                self.misses += 1
                reward -= 5 # Miss penalty
                missed = True
                # SFX: Failure buzz
                self._create_particles(pygame.Vector2(self.player_pos.x, self.GROUND_Y), 30, self.COLOR_MISS, 'miss')

            # Reset jump state regardless of outcome
            self.is_jumping = False
            self.player_pos.y = self.GROUND_Y
            self.player_vel.y = 0
            
        return reward, missed

    def _update_platforms(self):
        for plat in self.platforms:
            cycle_pos = (self.steps + plat["cycle_offset"]) % self.PLATFORM_CYCLE_FRAMES
            plat["is_visible"] = cycle_pos < self.PLATFORM_VISIBLE_DURATION

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _check_termination(self):
        return (
            self.score >= self.WIN_SCORE or
            self.misses >= self.MAX_MISSES
        )

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, 3), but we need (height, width, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
        }

    def _render_background(self):
        # Efficient gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_elements(self):
        # Render particles first (behind other elements)
        for p in self.particles:
            p.draw(self.screen)

        # Render platforms
        for plat in self.platforms:
            if plat["is_visible"]:
                # Pulsating glow effect
                cycle_pos = (self.steps + plat["cycle_offset"]) % self.PLATFORM_CYCLE_FRAMES
                if cycle_pos < self.PLATFORM_VISIBLE_DURATION:
                    # Glow is strongest at the start and end of visibility
                    pulse = abs(cycle_pos - self.PLATFORM_VISIBLE_DURATION / 2) / (self.PLATFORM_VISIBLE_DURATION / 2)
                    glow_size = int(8 * pulse)
                    glow_rect = plat["rect"].inflate(glow_size, glow_size)
                    
                    # Use a surface for alpha blending the glow
                    glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                    glow_alpha = 100 + 100 * pulse
                    pygame.draw.rect(glow_surface, (*self.COLOR_PLATFORM_GLOW[:3], glow_alpha), (0,0, *glow_rect.size), border_radius=8)
                    self.screen.blit(glow_surface, glow_rect.topleft)

                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat["rect"], border_radius=4)
        
        # Render player
        player_rect = pygame.Rect(
            self.player_pos.x - self.PLAYER_SIZE[0] / 2,
            self.player_pos.y - self.PLAYER_SIZE[1],
            *self.PLAYER_SIZE
        )
        
        # Squash and stretch based on velocity
        squash = min(5, max(-10, self.player_vel.y))
        player_render_rect = player_rect.inflate(-squash, squash)

        # Player glow
        glow_radius = 25 + 5 * math.sin(self.steps * 0.2)
        pygame.gfxdraw.filled_circle(self.screen, int(player_render_rect.centerx), int(player_render_rect.centery), int(glow_radius), self.COLOR_PLAYER_GLOW)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_render_rect, border_radius=4)
        
        # Render jump selector
        if not self.is_jumping:
            for i in range(len(self.JUMP_STRENGTHS)):
                color = self.COLOR_PLAYER if i <= self.selected_jump_index else (100, 100, 0)
                y_offset = i * 8
                bar_width = 10 + i * 5
                pygame.draw.rect(self.screen, color, (player_rect.centerx - bar_width/2, player_rect.top - 15 - y_offset, bar_width, 4), border_radius=2)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Misses
        for i in range(self.MAX_MISSES):
            color = self.COLOR_MISS if i < self.misses else (100, 100, 100)
            x_pos = self.WIDTH - 40 - (i * 40)
            pygame.draw.line(self.screen, color, (x_pos, 20), (x_pos + 20, 40), 4)
            pygame.draw.line(self.screen, color, (x_pos + 20, 20), (x_pos, 40), 4)
            
        # Ground line
        pygame.draw.line(self.screen, self.COLOR_PLATFORM, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 2)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0, 180))
            self.screen.blit(overlay, (0,0))
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, position, count, color, p_type):
        for _ in range(count):
            if p_type == 'land':
                vel = pygame.Vector2(random.uniform(-2, 2), random.uniform(-4, -1))
                lifespan = random.randint(20, 40)
                radius = random.uniform(3, 7)
                self.particles.append(self.Particle(position, vel, radius, color, lifespan, gravity=0.1))
            elif p_type == 'miss':
                vel = pygame.Vector2(random.uniform(-3, 3), random.uniform(-2, 2))
                lifespan = random.randint(30, 50)
                radius = random.uniform(4, 8)
                self.particles.append(self.Particle(position, vel, radius, color, lifespan, gravity=0.2))
            elif p_type == 'jump':
                vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(1, 3))
                lifespan = random.randint(10, 20)
                radius = random.uniform(2, 5)
                self.particles.append(self.Particle(position, vel, radius, color, lifespan, gravity=0.05))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Ensure the display is properly initialized for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    total_reward = 0
    
    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Rhythmic Platformer")
    clock = pygame.time.Clock()
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    print("--- Controls ---")
    print("Up/Down Arrows: Select jump height")
    print("Spacebar: Jump")
    print("R: Reset game")
    print("Q: Quit")
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    total_reward = 0
                    done = False
                
        # --- Action Polling ---
        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        else:
            movement = 0
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # --- Frame Rate ---
        clock.tick(GameEnv.FPS)
        
    env.close()