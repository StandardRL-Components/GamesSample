
# Generated: 2025-08-27T13:35:56.205163
# Source Brief: brief_00420.md
# Brief Index: 420

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to run, and ↑ or Space to jump. "
        "Reach the green flag as fast as you can!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling platformer. Guide your robot through a "
        "procedurally generated level, avoiding pitfalls to reach the goal. "
        "Your score is based on your completion time."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.LEVEL_WIDTH = 6400  # 10x screen width
        self.FPS = 30
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_PLAYER = (60, 160, 255)
        self.COLOR_PLAYER_EYE = (255, 255, 255)
        self.COLOR_PLATFORM = (120, 130, 140)
        self.COLOR_PLATFORM_TOP = (150, 160, 170)
        self.COLOR_FLAG_POLE = (200, 200, 200)
        self.COLOR_FLAG = (80, 220, 100)
        self.COLOR_UI = (240, 240, 240)
        self.COLOR_JUMP_PARTICLE = (100, 200, 255)
        self.COLOR_LAND_PARTICLE = (200, 200, 200)

        # Physics
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15
        self.MOVE_SPEED = 6.0
        self.FRICTION = 0.85
        
        # Player
        self.PLAYER_WIDTH, self.PLAYER_HEIGHT = 24, 32

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 72)
        
        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_on_ground = False
        self.player_last_y = 0.0
        self.platforms = []
        self.particles = []
        self.scroll_x = 0.0
        self.platform_max_gap = 100.0
        self.visited_platforms = set()
        
        self.reset()

        # Final validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self._generate_platforms()
        
        start_platform = self.platforms[0]
        self.player_pos = np.array([start_platform.centerx, start_platform.top - self.PLAYER_HEIGHT], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_on_ground = False
        self.player_last_y = self.player_pos[1]
        
        self.particles = []
        self.scroll_x = 0.0
        self.platform_max_gap = 100.0
        self.visited_platforms = {0}
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(self.FPS)
        
        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] = -self.MOVE_SPEED
        elif movement == 4:  # Right
            self.player_vel[0] = self.MOVE_SPEED
        
        # Jump
        if (movement == 1 or space_held) and self.player_on_ground:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.player_on_ground = False
            # Sound: Player Jump
            self._create_particles(20, self.player_pos + [0, self.PLAYER_HEIGHT], self.COLOR_JUMP_PARTICLE, 'up')

        # --- Physics and State Update ---
        self._update_player(movement)
        self._update_particles()
        
        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 500 == 0:
            self.platform_max_gap *= 1.05

        # --- Scoring and Termination ---
        self.steps += 1
        reward = 0.1  # Survival reward
        
        # Check for landing on new platforms
        player_rect = self._get_player_rect()
        for i, plat in enumerate(self.platforms):
            if player_rect.colliderect(plat) and self.player_vel[1] >= 0:
                if i not in self.visited_platforms:
                    self.visited_platforms.add(i)
                    reward += 1.0
                    self.score += 1.0
                    # Sound: New Platform Land
                    self._create_particles(10, [player_rect.midbottomx, player_rect.bottom], self.COLOR_LAND_PARTICLE, 'side')
                    break
        
        terminated = False
        if self.player_pos[1] > self.HEIGHT + 50:  # Fell off screen
            terminated = True
            reward = -100.0
            self.score -= 100
        elif self.player_pos[0] >= self.LEVEL_WIDTH - 50:  # Reached flag
            terminated = True
            time_bonus = max(0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS)
            win_reward = 100.0 * time_bonus
            reward += win_reward
            self.score += win_reward
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        # Apply friction
        if movement not in [3, 4]:
            self.player_vel[0] *= self.FRICTION

        # Apply gravity
        self.player_vel[1] += self.GRAVITY

        # Store last position for collision detection
        self.player_last_y = self.player_pos[1]

        # Update position
        self.player_pos += self.player_vel

        # Clamp horizontal velocity
        self.player_vel[0] = np.clip(self.player_vel[0], -self.MOVE_SPEED, self.MOVE_SPEED)

        # Prevent moving off left edge
        if self.player_pos[0] < self.PLAYER_WIDTH / 2:
            self.player_pos[0] = self.PLAYER_WIDTH / 2
            self.player_vel[0] = 0

        # Collision detection with platforms
        self.player_on_ground = False
        player_rect = self._get_player_rect()
        
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # Check if player was above the platform in the previous frame
                if self.player_vel[1] > 0 and (self.player_last_y + self.PLAYER_HEIGHT) <= plat.top + 1:
                    self.player_pos[1] = plat.top - self.PLAYER_HEIGHT
                    self.player_vel[1] = 0
                    self.player_on_ground = True
                    break
    
    def _generate_platforms(self):
        self.platforms = []
        
        # Start platform
        plat_y = self.HEIGHT - 50
        start_plat = pygame.Rect(50, plat_y, 200, 20)
        self.platforms.append(start_plat)
        
        current_x = start_plat.right
        
        while current_x < self.LEVEL_WIDTH - 200:
            gap = self.np_random.uniform(40, self.platform_max_gap)
            current_x += gap
            
            width = self.np_random.uniform(80, 250)
            
            # Ensure y stays within a jumpable range
            y_change = self.np_random.uniform(-80, 80)
            plat_y = np.clip(plat_y + y_change, 150, self.HEIGHT - 50)
            
            new_plat = pygame.Rect(int(current_x), int(plat_y), int(width), 20)
            self.platforms.append(new_plat)
            
            current_x += width

    def _get_player_rect(self):
        return pygame.Rect(
            self.player_pos[0] - self.PLAYER_WIDTH / 2,
            self.player_pos[1],
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT
        )

    def _get_observation(self):
        # Update camera scroll
        target_scroll_x = self.player_pos[0] - self.WIDTH / 2
        self.scroll_x += (target_scroll_x - self.scroll_x) * 0.1
        self.scroll_x = max(0, min(self.scroll_x, self.LEVEL_WIDTH - self.WIDTH))

        # --- Render All Elements ---
        self._render_background()
        self._render_game_elements()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Parallax grid
        grid_offset = -self.scroll_x * 0.5
        for i in range(0, self.WIDTH + 100, 50):
            x = int(i + grid_offset % 50)
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT + 100, 50):
            y = int(i)
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_game_elements(self):
        # Draw platforms
        for plat in self.platforms:
            screen_plat = plat.move(-self.scroll_x, 0)
            if screen_plat.right > 0 and screen_plat.left < self.WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_plat)
                top_rect = pygame.Rect(screen_plat.left, screen_plat.top, screen_plat.width, 4)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, top_rect)
        
        # Draw flag
        flag_pole_x = self.LEVEL_WIDTH - 50 - self.scroll_x
        if 0 < flag_pole_x < self.WIDTH:
            flag_base_y = self.platforms[-1].top
            pygame.draw.line(self.screen, self.COLOR_FLAG_POLE, (flag_pole_x, flag_base_y), (flag_pole_x, flag_base_y - 80), 3)
            flag_points = [
                (flag_pole_x, flag_base_y - 80),
                (flag_pole_x + 40, flag_base_y - 70),
                (flag_pole_x, flag_base_y - 60)
            ]
            pygame.gfxdraw.aapolygon(self.screen, flag_points, self.COLOR_FLAG)
            pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_FLAG)

    def _render_player(self):
        player_rect_on_screen = self._get_player_rect().move(-self.scroll_x, 0)
        
        # Squash and stretch based on vertical velocity
        squash = min(5, max(-5, self.player_vel[1] * 0.5))
        body_height = self.PLAYER_HEIGHT - squash
        body_width = self.PLAYER_WIDTH + squash
        
        # Bobbing animation for running
        bob = math.sin(self.steps * 0.5) * 2 if self.player_on_ground and abs(self.player_vel[0]) > 0.1 else 0
        
        body_rect = pygame.Rect(
            player_rect_on_screen.centerx - body_width / 2,
            player_rect_on_screen.bottom - body_height + bob,
            body_width,
            body_height
        )
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=4)
        
        # Eye
        eye_dir = 1 if self.player_vel[0] >= 0 else -1
        eye_pos = (int(body_rect.centerx + eye_dir * 5), int(body_rect.centery - 5))
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_EYE, eye_pos, 3)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0] - self.scroll_x), int(p['pos'][1]))
            radius = int(p['life'] / p['max_life'] * p['size'])
            if radius > 0:
                pygame.draw.circle(self.screen, p['color'], pos, radius)

    def _render_ui(self):
        # Timer
        time_text = f"Time: {self.steps / self.FPS:.2f}s"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI)
        self.screen.blit(time_surf, (10, 10))

        # Score (based on platforms)
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 35))

        if self.game_over:
            msg = "SUCCESS!" if self.player_pos[0] >= self.LEVEL_WIDTH - 50 else "GAME OVER"
            msg_surf = self.font_big.render(msg, True, self.COLOR_UI)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos.tolist(),
            "player_vel": self.player_vel.tolist(),
        }
    
    def _create_particles(self, count, pos, color, style):
        for _ in range(count):
            if style == 'up':
                vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(1, 4)]
            elif style == 'side':
                vel = [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 0)]
            else: # explosion
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 5)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]

            life = self.np_random.integers(15, 25)
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.uniform(3, 6)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # particle gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

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

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Change to 'windows' or 'mac' if needed, or remove for default.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robot Platformer")
    
    terminated = False
    
    # --- Key mapping ---
    # MultiDiscrete([5, 2, 2])
    # [0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    # [1]: Space button (0=released, 1=held)
    # [2]: Shift button (0=released, 1=held)
    action = np.array([0, 0, 0])

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Buttons
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)


    env.close()