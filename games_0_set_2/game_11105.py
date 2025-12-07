import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    GameEnv: Synchronized Platform Jumping
    
    An arcade-style game where the agent controls two platforms. The goal is to
    make them jump in sync to build a score multiplier and reach a target height.
    
    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Jump Command (0: No-op, 1: Low Jump, 2: High Jump)
    - action[1]: Unused
    - action[2]: Jump Modifier (0: Symmetric, 1: Asymmetric)
    
    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    
    Reward Structure:
    - +0.1 for a synced jump (both platforms jump the same height).
    - -0.1 for a desynced jump.
    - +1.0 for increasing the score multiplier.
    - +100 for winning (reaching the target height).
    - -100 for losing (a platform falls off the screen, which is now impossible).
    - Timeout: Episode steps >= MAX_STEPS.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control two platforms and make them jump in sync to build a score "
        "multiplier and reach a target height."
    )
    user_guide = (
        "Controls: ↑ for a high jump, ↓ for a low jump. "
        "Hold Shift to make the jump asymmetric."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_HEIGHT = 150.0
    MAX_STEPS = 1000

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (40, 60, 100)
    COLOR_P1 = (255, 70, 70)
    COLOR_P2 = (70, 120, 255)
    COLOR_TARGET_LINE = (100, 255, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_SYNC = (255, 255, 100)
    COLOR_DESYNC = (100, 100, 120)

    # Physics
    GRAVITY = -0.3
    LOW_JUMP_STRENGTH = 6.0
    HIGH_JUMP_STRENGTH = 9.0
    
    # Visuals
    PLATFORM_WIDTH = 80
    PLATFORM_HEIGHT = 15
    CAMERA_SMOOTHING = 0.08

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_multiplier = pygame.font.SysFont("Consolas", 48, bold=True)

        self.p1 = None
        self.p2 = None
        self.particles = []
        self.camera_y = 0.0
        self.steps = 0
        self.score = 0
        self.multiplier = 1.0
        
        self.bg_surface = self._create_gradient_background()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        start_y = 0.0  # Start on the ground
        self.p1 = self._Platform(self.SCREEN_WIDTH * 0.35, start_y, self.COLOR_P1)
        self.p2 = self._Platform(self.SCREEN_WIDTH * 0.65, start_y, self.COLOR_P2)
        self.p1.is_grounded = True
        self.p2.is_grounded = True
        
        self.particles = []
        self.camera_y = start_y
        self.steps = 0
        self.score = 0
        self.multiplier = 1.0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        
        # 1. Handle actions to determine jump strengths
        jump1_strength, jump2_strength = self._handle_actions(action)
        
        # A jump is only possible if both platforms are on the ground
        # and a jump action was requested.
        if self.p1.is_grounded and self.p2.is_grounded and jump1_strength > 0:
            is_synced = jump1_strength == jump2_strength
            
            # 2. Apply jump impulses
            self.p1.jump(jump1_strength)
            self.p2.jump(jump2_strength)

            # 3. Create visual feedback for jumps
            self._create_jump_particles(self.p1)
            self._create_jump_particles(self.p2)
            
            # 4. Update multiplier and calculate related rewards
            old_multiplier = self.multiplier
            if is_synced:
                self.multiplier = min(5.0, self.multiplier + 0.25)
                reward += 0.1
                if self.multiplier > old_multiplier:
                    reward += 1.0
                    self._create_sync_desync_particles(True)
            else:
                self.multiplier = max(1.0, self.multiplier - 0.5)
                reward -= 0.1
                self._create_sync_desync_particles(False)

        # 5. Update physics
        self._update_physics()

        # 6. Check for termination conditions
        terminated, won = self._check_termination()

        # 7. Calculate terminal rewards
        if terminated and won:
            reward += 100
        
        # Truncation for timeout
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        command = action[0]       # 0: no-op, 1: low jump, 2: high jump
        is_asymmetric = action[2] == 1

        if command == 0:
            return 0, 0

        base_jump = self.LOW_JUMP_STRENGTH if command == 1 else self.HIGH_JUMP_STRENGTH
        
        if is_asymmetric:
            # Asymmetric jump: one platform gets the opposite of the base jump
            jump1 = self.LOW_JUMP_STRENGTH if base_jump == self.HIGH_JUMP_STRENGTH else self.HIGH_JUMP_STRENGTH
            jump2 = base_jump
        else:
            # Symmetric jump: both platforms get the base jump
            jump1 = base_jump
            jump2 = base_jump
            
        return jump1, jump2

    def _update_physics(self):
        self.p1.update(self.GRAVITY)
        self.p2.update(self.GRAVITY)
        
        target_camera_y = (self.p1.y + self.p2.y) / 2
        self.camera_y += (target_camera_y - self.camera_y) * self.CAMERA_SMOOTHING

        for p in self.particles[:]:
            p.update(self.GRAVITY)
            if p.lifespan <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        avg_height = (self.p1.y + self.p2.y) / 2
        
        if avg_height >= self.TARGET_HEIGHT:
            return True, True
            
        # Lose condition removed as platforms can no longer fall through the floor
        # if self.p1.y < 0 or self.p2.y < 0:
        #     return True, False
            
        if self.steps >= self.MAX_STEPS:
            return True, False
            
        return False, False

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        avg_height = (self.p1.y + self.p2.y) / 2 if self.p1 and self.p2 else 0
        return {
            "score": self.score,
            "steps": self.steps,
            "height": avg_height,
            "multiplier": self.multiplier,
        }

    def _render_all(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game_elements()
        self._render_ui()

    def _world_to_screen(self, x, y):
        screen_x = int(x)
        screen_y = int(self.SCREEN_HEIGHT / 2 - (y - self.camera_y))
        return screen_x, screen_y

    def _render_game_elements(self):
        tx, ty = self._world_to_screen(0, self.TARGET_HEIGHT)
        pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (0, ty), (self.SCREEN_WIDTH, ty), 2)

        for p in self.particles:
            p.draw(self.screen, self._world_to_screen)

        self.p1.draw(self.screen, self._world_to_screen)
        self.p2.draw(self.screen, self._world_to_screen)
        
        gx, gy = self._world_to_screen(0, 0)
        if 0 <= gy < self.SCREEN_HEIGHT:
            pygame.draw.line(self.screen, (150, 150, 160), (0, gy), (self.SCREEN_WIDTH, gy), 3)

    def _render_ui(self):
        avg_height = (self.p1.y + self.p2.y) / 2
        height_text = self.font_main.render(f"Height: {max(0, avg_height):.1f}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (10, 10))

        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 40))

        multiplier_text = self.font_multiplier.render(f"{self.multiplier:.2f}x", True, self.COLOR_TEXT)
        text_rect = multiplier_text.get_rect(center=(self.SCREEN_WIDTH / 2, 40))
        self.screen.blit(multiplier_text, text_rect)

        steps_text = self.font_main.render(f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        steps_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(steps_text, steps_rect)

    def _create_gradient_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio)
                for i in range(3)
            ]
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _create_jump_particles(self, platform):
        for _ in range(15):
            angle = random.uniform(math.pi * 1.25, math.pi * 1.75)
            speed = random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed * 0.5)
            pos = (platform.x, platform.y - self.PLATFORM_HEIGHT / 2)
            lifespan = random.randint(15, 30)
            radius = random.uniform(1, 4)
            self.particles.append(self._Particle(pos, vel, platform.color, radius, lifespan))
            
    def _create_sync_desync_particles(self, is_synced):
        pos_x = (self.p1.x + self.p2.x) / 2
        pos_y = (self.p1.y + self.p2.y) / 2
        color = self.COLOR_SYNC if is_synced else self.COLOR_DESYNC
        count = 30 if is_synced else 20
        
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5) if is_synced else random.uniform(0.5, 2)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            pos = (pos_x, pos_y)
            lifespan = random.randint(20, 40)
            radius = random.uniform(2, 5) if is_synced else random.uniform(1, 3)
            self.particles.append(self._Particle(pos, vel, color, radius, lifespan, has_gravity=not is_synced))

    def close(self):
        pygame.quit()

    class _Platform:
        def __init__(self, x, y, color):
            self.x = x
            self.y = y
            self.vy = 0.0
            self.color = color
            self.width = GameEnv.PLATFORM_WIDTH
            self.height = GameEnv.PLATFORM_HEIGHT
            self.squash_timer = 0
            self.squash_amount = 0
            self.is_grounded = False

        def jump(self, strength):
            if strength > 0:
                self.vy = strength
                self.squash_timer = 10
                self.squash_amount = strength / GameEnv.HIGH_JUMP_STRENGTH
                self.is_grounded = False

        def update(self, gravity):
            self.vy += gravity
            self.y += self.vy
            
            if self.y <= 0:
                self.y = 0
                self.vy = 0
                self.is_grounded = True
            else:
                self.is_grounded = False

            if self.squash_timer > 0:
                self.squash_timer -= 1
        
        def draw(self, surface, world_to_screen):
            squash_factor = 0
            if self.squash_timer > 0:
                progress = self.squash_timer / 10
                squash_factor = math.sin(progress * math.pi) * self.squash_amount

            current_width = self.width * (1 + 0.3 * squash_factor)
            current_height = self.height * (1 - 0.3 * squash_factor)
            
            for i in range(4, 0, -1):
                glow_alpha = 40 - i * 8
                glow_size_w = current_width + i * 4
                glow_size_h = current_height + i * 4
                
                glow_x = self.x - glow_size_w / 2
                glow_y = self.y + glow_size_h / 2
                
                sx, sy = world_to_screen(glow_x, glow_y)
                rect = pygame.Rect(sx, sy, glow_size_w, glow_size_h)
                
                glow_color = (*self.color, glow_alpha)
                shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(shape_surf, glow_color, shape_surf.get_rect(), border_radius=5)
                surface.blit(shape_surf, rect)

            rect_x = self.x - current_width / 2
            rect_y = self.y + current_height / 2
            sx, sy = world_to_screen(rect_x, rect_y)
            
            pygame.draw.rect(surface, self.color, (sx, sy, current_width, current_height), border_radius=5)
            pygame.draw.rect(surface, tuple(min(255, c+50) for c in self.color), (sx, sy, current_width, current_height), 2, border_radius=5)

    class _Particle:
        def __init__(self, pos, vel, color, radius, lifespan, has_gravity=True):
            self.x, self.y = pos
            self.vx, self.vy = vel
            self.color = color
            self.radius = radius
            self.lifespan = lifespan
            self.max_lifespan = lifespan
            self.has_gravity = has_gravity

        def update(self, gravity):
            if self.has_gravity:
                self.vy += gravity * 0.5
            self.x += self.vx
            self.y += self.vy
            self.lifespan -= 1

        def draw(self, surface, world_to_screen):
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            color = (*self.color, alpha)
            sx, sy = world_to_screen(self.x, self.y)
            
            if self.radius > 0:
                pygame.gfxdraw.aacircle(surface, sx, sy, int(self.radius), color)
                pygame.gfxdraw.filled_circle(surface, sx, sy, int(self.radius), color)

if __name__ == '__main__':
    # --- Manual Play Example ---
    # This example is updated to reflect the new action mapping.
    env = GameEnv()
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Synchronized Platform Jumper")
    clock = pygame.time.Clock()

    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    print("----------------\n")
    
    running = True
    while running:
        action = [0, 0, 0] # [command, unused, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False

        # Continuous key state polling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 2  # High jump command
        elif keys[pygame.K_DOWN]:
            action[0] = 1  # Low jump command
        
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Final Height: {info['height']:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS for smooth viewing

    env.close()