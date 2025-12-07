import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ←→ to jump left/right, and ↑ to jump straight up. "
        "Hold Space for a higher jump. Hold Shift for a shorter hop."
    )

    game_description = (
        "Hop between procedurally generated neon platforms to reach the top. "
        "Falling off the screen costs a life. Reach platform 100 to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.WIN_PLATFORM = 100

        # Physics constants
        self.GRAVITY = 0.4
        self.PLAYER_HORIZ_SPEED = 6.0
        self.JUMP_POWER_SHORT = 6.0
        self.JUMP_POWER_MEDIUM = 9.0
        self.JUMP_POWER_HIGH = 12.0
        self.AIR_FRICTION = 0.95
        self.PLAYER_WIDTH, self.PLAYER_HEIGHT = 20, 20
        
        # Color constants
        self.COLOR_BG_TOP = (13, 13, 38)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_PLAYER = (77, 255, 77)
        self.COLOR_PLAYER_GLOW = (77, 255, 77, 50)
        self.COLOR_PLATFORM_NORMAL = (255, 51, 204)
        self.COLOR_PLATFORM_UNSTABLE = (255, 255, 51)
        self.COLOR_TEXT = (240, 240, 255)
        self.COLOR_PARTICLE_JUMP = (200, 200, 255)
        self.COLOR_PARTICLE_LAND = self.COLOR_PLATFORM_NORMAL
        self.COLOR_PARTICLE_FALL = (255, 50, 50)

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Internal state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.lives = 0
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.platforms = []
        self.particles = []
        self.camera_y = 0
        self.highest_platform_id = 0
        self.player_squash = 0.0
        self.np_random = None

        # Initialize and validate
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        
        self.platforms = []
        self._generate_platforms(count=15, y_start=self.SCREEN_HEIGHT - 20, is_initial=True)
        
        start_platform = self.platforms[0]
        self.player_pos = pygame.Vector2(start_platform['rect'].centerx, start_platform['rect'].top - self.PLAYER_HEIGHT)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = True
        
        self.particles = []
        self.camera_y = 0
        self.highest_platform_id = 0
        self.player_squash = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.clock.tick(self.FPS)
        self.steps += 1
        self.reward_this_step = 0

        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._check_collisions()
            self._update_platforms()
            self._update_camera()
        
        self._update_particles()
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        reward = self.reward_this_step

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if not self.on_ground:
            self.reward_this_step -= 0.02 # Small penalty for no-op in air
            return

        jump_power = self.JUMP_POWER_MEDIUM
        if space_held:
            jump_power = self.JUMP_POWER_HIGH
        elif shift_held:
            jump_power = self.JUMP_POWER_SHORT

        vert_vel, horiz_vel = 0, 0

        if movement == 1:  # Up
            vert_vel = -jump_power
            self.reward_this_step += 0.1
        elif movement == 2:  # Down
            vert_vel = jump_power * 0.5  # Hop down
            self.reward_this_step -= 0.02
        elif movement == 3:  # Left
            vert_vel = -jump_power * 0.7
            horiz_vel = -self.PLAYER_HORIZ_SPEED
            self.reward_this_step += 0.1 # Upward component
        elif movement == 4:  # Right
            vert_vel = -jump_power * 0.7
            horiz_vel = self.PLAYER_HORIZ_SPEED
            self.reward_this_step += 0.1 # Upward component
        else: # No-op
            self.reward_this_step -= 0.02

        if movement != 0:
            self.player_vel.y = vert_vel
            self.player_vel.x = horiz_vel
            self.on_ground = False
            self.player_squash = 0.3 # Start jump animation
            self._create_particles(self.player_pos + pygame.Vector2(self.PLAYER_WIDTH / 2, self.PLAYER_HEIGHT), 10, self.COLOR_PARTICLE_JUMP, 2)

    def _update_player(self):
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
        
        self.player_vel.x *= self.AIR_FRICTION
        self.player_pos += self.player_vel

        # Screen bounds
        if self.player_pos.x < 0:
            self.player_pos.x = 0
            self.player_vel.x *= -0.5
        if self.player_pos.x > self.SCREEN_WIDTH - self.PLAYER_WIDTH:
            self.player_pos.x = self.SCREEN_WIDTH - self.PLAYER_WIDTH
            self.player_vel.x *= -0.5
        
        # Fall detection
        if self.player_pos.y > self.camera_y + self.SCREEN_HEIGHT + 50:
            self.lives -= 1
            self.reward_this_step -= 5.0
            if self.lives > 0:
                self._reset_player_position()
            else:
                self.game_over = True

    def _reset_player_position(self):
        # Find the highest platform the player has landed on
        target_platform = None
        for p in self.platforms:
            if p['id'] == self.highest_platform_id:
                target_platform = p['rect']
                break
        if target_platform is None:
            target_platform = self.platforms[0]['rect']

        self.player_pos = pygame.Vector2(target_platform.centerx, target_platform.top - self.PLAYER_HEIGHT)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = True
        self._create_particles(self.player_pos + pygame.Vector2(self.PLAYER_WIDTH / 2, self.PLAYER_HEIGHT), 30, self.COLOR_PARTICLE_FALL, 4)

    def _check_collisions(self):
        if self.player_vel.y < 0:
            return  # Can't land while moving up

        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        
        for plat in self.platforms:
            if player_rect.colliderect(plat['rect']):
                # Check if player was above the platform in the previous frame
                prev_player_bottom = self.player_pos.y + self.PLAYER_HEIGHT - self.player_vel.y
                if prev_player_bottom <= plat['rect'].top:
                    self.player_pos.y = plat['rect'].top - self.PLAYER_HEIGHT
                    self.player_vel.y = 0
                    self.player_vel.x = 0
                    self.on_ground = True
                    self.player_squash = -0.5 # Start land animation
                    
                    land_color = self.COLOR_PLATFORM_UNSTABLE if plat['type'] == 'unstable' else self.COLOR_PLATFORM_NORMAL
                    self._create_particles(self.player_pos + pygame.Vector2(self.PLAYER_WIDTH / 2, self.PLAYER_HEIGHT), 15, land_color, 1.5)

                    if plat['id'] > self.highest_platform_id:
                        self.reward_this_step += 1.0
                        self.highest_platform_id = plat['id']
                        self.score = self.highest_platform_id
                        if self.score >= self.WIN_PLATFORM:
                            self.reward_this_step += 100
                            self.game_over = True
                    break

    def _update_platforms(self):
        # Remove platforms far below the camera
        self.platforms = [p for p in self.platforms if p['rect'].top < self.camera_y + self.SCREEN_HEIGHT + 100]

        # Generate new platforms if needed
        highest_y = min(p['rect'].y for p in self.platforms) if self.platforms else self.SCREEN_HEIGHT
        if highest_y > self.camera_y - self.SCREEN_HEIGHT:
             self._generate_platforms(count=10, y_start=highest_y)

    def _generate_platforms(self, count, y_start, is_initial=False):
        plat_y = y_start
        last_x = self.SCREEN_WIDTH / 2

        for i in range(count):
            plat_id = self.platforms[-1]['id'] + 1 if self.platforms else 0

            # Difficulty scaling
            difficulty_factor = max(0, self.score - 10)
            width_reduction = 0.05 * (difficulty_factor // 10)
            gap_increase = 0.02 * (difficulty_factor // 10)
            
            min_width = 40
            base_width = 120 if is_initial else 100
            plat_width = max(min_width, base_width * (1 - width_reduction))

            max_gap_x = 120 + 80 * gap_increase
            max_gap_y = 100 + 50 * gap_increase
            
            if is_initial and i == 0:
                plat_x = self.SCREEN_WIDTH / 2 - 100
                plat_width = 200
            else:
                plat_x = last_x + self.np_random.uniform(-max_gap_x, max_gap_x)

            plat_x = np.clip(plat_x, 0, self.SCREEN_WIDTH - plat_width)
            plat_y -= self.np_random.uniform(30, max_gap_y)
            
            # Unstable platform chance
            plat_type = 'normal'
            if self.score > 20:
                unstable_chance = 0.01 * ((self.score - 20) // 5)
                if self.np_random.random() < unstable_chance:
                    plat_type = 'unstable'

            new_plat = {
                'rect': pygame.Rect(int(plat_x), int(plat_y), int(plat_width), 20),
                'type': plat_type,
                'id': plat_id
            }
            self.platforms.append(new_plat)
            last_x = plat_x

    def _update_camera(self):
        target_y = self.player_pos.y - self.SCREEN_HEIGHT * 0.6
        # Camera only moves up, never down, increasing tension
        new_camera_y = min(self.camera_y, target_y)
        self.camera_y += (new_camera_y - self.camera_y) * 0.1
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] = max(0, p['lifespan'] * 0.2)

    def _create_particles(self, pos, count, color, speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed_rand = self.np_random.uniform(speed * 0.5, speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed_rand
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.integers(3, 6)
            })

    def _check_termination(self):
        return self.game_over

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        self._render_background()
        self._render_platforms()
        self._render_particles()
        self._render_player()
        self._render_ui()

    def _render_background(self):
        # This is a fast way to draw a vertical gradient
        self.screen.fill(self.COLOR_BG_BOTTOM)
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_platforms(self):
        for plat in self.platforms:
            rect = plat['rect'].copy()
            rect.y -= int(self.camera_y)
            
            color = self.COLOR_PLATFORM_NORMAL
            if plat['type'] == 'unstable':
                # Flickering effect for unstable platforms
                if (self.steps // 3) % 2 == 0:
                    color = self.COLOR_PLATFORM_UNSTABLE
                else:
                    color = tuple(c // 2 for c in self.COLOR_PLATFORM_UNSTABLE)

            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            # Inner darker part for depth
            inner_rect = rect.inflate(-6, -6)
            inner_color = tuple(c * 0.7 for c in color)
            pygame.draw.rect(self.screen, inner_color, inner_rect, border_radius=3)

    def _render_player(self):
        # Squash and stretch animation
        self.player_squash *= 0.8
        squash_factor = math.sin(self.player_squash)
        
        width = self.PLAYER_WIDTH * (1 + squash_factor * 0.5)
        height = self.PLAYER_HEIGHT * (1 - squash_factor * 0.5)
        
        pos_x = self.player_pos.x - (width - self.PLAYER_WIDTH) / 2
        pos_y = self.player_pos.y - self.camera_y + (self.PLAYER_HEIGHT - height)

        player_rect = pygame.Rect(int(pos_x), int(pos_y), int(width), int(height))

        # Glow effect
        glow_rect = player_rect.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_particles(self):
        for p in self.particles:
            pos = p['pos'] - pygame.Vector2(0, self.camera_y)
            alpha = int(255 * (p['lifespan'] / 40))
            color = (*p['color'], alpha)
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(p['radius']), color)
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(p['radius']), color)
            except OverflowError: # Can happen if particle radius becomes huge/negative
                pass

    def _render_ui(self):
        # Score (Platforms Reached)
        score_text = self.font_ui.render(f"SCORE: {self.score}/{self.WIN_PLATFORM}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (10, 10))

        # Timer/Steps
        timer_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH // 2 - timer_text.get_width() // 2, 10))

        if self.game_over:
            if self.score >= self.WIN_PLATFORM:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            over_text = self.font_game_over.render(msg, True, self.COLOR_PLAYER)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            # Text shadow
            shadow_text = self.font_game_over.render(msg, True, self.COLOR_BG_TOP)
            shadow_rect = shadow_text.get_rect(center=(self.SCREEN_WIDTH // 2 + 3, self.SCREEN_HEIGHT // 2 + 3))
            self.screen.blit(shadow_text, shadow_rect)
            self.screen.blit(over_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "highest_platform": self.highest_platform_id,
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == "__main__":
    # To run with display, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need to create a window
    pygame.display.set_caption("Neon Jumper")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    obs, info = env.reset()
    done = False
    
    # Mapping keys to actions for human play
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    running = True
    while running:
        movement_action = 0
        space_action = 0
        shift_action = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        
        for key, move in key_to_action.items():
            if keys[key]:
                movement_action = move
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Score: {info['score']}, Steps: {info['steps']}")
            # Wait a moment before auto-resetting for human play
            pygame.time.wait(2000)
            obs, info = env.reset()

    env.close()