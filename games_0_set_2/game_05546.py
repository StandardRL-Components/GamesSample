import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Use arrow keys (↑↓←→) to move. Collect yellow souls and avoid the red ghouls."
    )

    game_description = (
        "Collect wandering souls in a haunted graveyard while evading relentless ghouls."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WIN_SOULS = 20
        self.MAX_TOUCHES = 4
        self.MAX_STEPS = 1500 # Increased from 1000 to allow more time
        self.PLAYER_SPEED = 4.0
        self.PLAYER_RADIUS = 10
        self.SOUL_RADIUS = 6
        self.GHOUL_RADIUS = 12
        self.GHOUL_SPEED = 1.5
        self.INVINCIBILITY_FRAMES = 90 # 3 seconds at 30fps

        # Colors
        self.COLOR_BG = (20, 15, 40)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_INVINCIBLE = (200, 200, 255)
        self.COLOR_SOUL = (255, 255, 0)
        self.COLOR_SOUL_GLOW = (255, 255, 100)
        self.COLOR_GHOUL = (220, 20, 60)
        self.COLOR_GHOUL_FLICKER = (180, 10, 40)
        self.COLOR_GRAVESTONE = (70, 80, 90)
        self.COLOR_TEXT_GOOD = (50, 255, 50)
        self.COLOR_TEXT_BAD = (255, 50, 50)
        self.COLOR_TEXT_MSG = (255, 255, 255)

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
        self.font_ui = pygame.font.Font(None, 36)
        self.font_msg = pygame.font.Font(None, 72)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.souls_collected = 0
        self.ghoul_touches = 0
        self.player_pos = None
        self.souls = []
        self.ghouls = []
        self.gravestones = []
        self.particles = []
        self.invincibility_timer = 0
        self.np_random = None

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.souls_collected = 0
        self.ghoul_touches = 0
        self.invincibility_timer = 0
        
        self.player_pos = np.array([50.0, self.SCREEN_HEIGHT / 2.0])
        
        self._generate_layout()
        self._spawn_souls(self.WIN_SOULS)
        self._spawn_ghouls(4)
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _generate_layout(self):
        self.gravestones = []
        # Create a border of "safe" space
        for i in range(5):
            x = self.np_random.integers(100, self.SCREEN_WIDTH - 100)
            y = self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
            w = self.np_random.integers(20, 60)
            h = self.np_random.integers(40, 80)
            self.gravestones.append(pygame.Rect(x, y, w, h))

    def _spawn_souls(self, num_souls):
        self.souls = []
        while len(self.souls) < num_souls:
            pos = np.array([
                self.np_random.uniform(20, self.SCREEN_WIDTH - 20),
                self.np_random.uniform(20, self.SCREEN_HEIGHT - 20)
            ])
            is_valid = True
            if math.hypot(pos[0] - self.player_pos[0], pos[1] - self.player_pos[1]) < 100:
                is_valid = False
            for stone in self.gravestones:
                if stone.collidepoint(pos):
                    is_valid = False
                    break
            if is_valid:
                self.souls.append(pos)

    def _spawn_ghouls(self, num_ghouls):
        self.ghouls = []
        paths = [
            # Horizontal patrol
            [np.array([100, 50], dtype=float), np.array([self.SCREEN_WIDTH - 100, 50], dtype=float)],
            # Vertical patrol
            [np.array([self.SCREEN_WIDTH - 50, 80], dtype=float), np.array([self.SCREEN_WIDTH - 50, self.SCREEN_HEIGHT - 80], dtype=float)],
            # Box patrol
            [np.array([100, 100], dtype=float), np.array([self.SCREEN_WIDTH - 150, 100], dtype=float), np.array([self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 100], dtype=float), np.array([100, self.SCREEN_HEIGHT - 100], dtype=float)],
            # Diagonal patrol
            [np.array([80, self.SCREEN_HEIGHT - 80], dtype=float), np.array([self.SCREEN_WIDTH - 80, 80], dtype=float)]
        ]
        for i in range(num_ghouls):
            path = paths[i % len(paths)]
            self.ghouls.append({
                "pos": np.copy(path[0]),
                "path": path,
                "target_idx": 1
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # Unpack action
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused
        
        self._update_player(movement)
        self._update_ghouls()
        self._update_particles()
        
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
        
        reward += self._handle_collisions()
        
        self.steps += 1
        self.score += reward

        terminated = self._check_termination()
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        if terminated or truncated:
            self.game_over = True
            if self.souls_collected >= self.WIN_SOULS:
                reward += 100
                self.score += 100
            elif self.ghoul_touches >= self.MAX_TOUCHES:
                reward -= 100
                self.score -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, movement):
        velocity = np.array([0.0, 0.0])
        if movement == 1: velocity[1] -= self.PLAYER_SPEED  # Up
        elif movement == 2: velocity[1] += self.PLAYER_SPEED  # Down
        elif movement == 3: velocity[0] -= self.PLAYER_SPEED  # Left
        elif movement == 4: velocity[0] += self.PLAYER_SPEED  # Right

        if np.any(velocity):
            new_pos = self.player_pos + velocity
            
            # World bounds collision
            new_pos[0] = np.clip(new_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
            new_pos[1] = np.clip(new_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

            # Gravestone collision
            player_rect = pygame.Rect(new_pos[0] - self.PLAYER_RADIUS, new_pos[1] - self.PLAYER_RADIUS, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
            for stone in self.gravestones:
                if stone.colliderect(player_rect):
                    # Simple stop, not perfect slide physics but good enough
                    return

            self.player_pos = new_pos

    def _update_ghouls(self):
        for ghoul in self.ghouls:
            target_pos = ghoul["path"][ghoul["target_idx"]]
            direction = target_pos - ghoul["pos"]
            distance = np.linalg.norm(direction)

            if distance < self.GHOUL_SPEED:
                ghoul["pos"] = np.copy(target_pos)
                ghoul["target_idx"] = (ghoul["target_idx"] + 1) % len(ghoul["path"])
            else:
                ghoul["pos"] += (direction / distance) * self.GHOUL_SPEED

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1

    def _handle_collisions(self):
        reward = 0
        
        # Player-Soul collision
        collected_souls = []
        for i, soul_pos in enumerate(self.souls):
            dist = np.linalg.norm(self.player_pos - soul_pos)
            if dist < self.PLAYER_RADIUS + self.SOUL_RADIUS:
                collected_souls.append(i)
                self.souls_collected += 1
                reward += 10
                # SFX: Soul collect sound

        # Remove collected souls in reverse to avoid index errors
        for i in sorted(collected_souls, reverse=True):
            del self.souls[i]

        # Player-Ghoul collision
        if self.invincibility_timer == 0:
            for ghoul in self.ghouls:
                dist = np.linalg.norm(self.player_pos - ghoul["pos"])
                if dist < self.PLAYER_RADIUS + self.GHOUL_RADIUS:
                    self.ghoul_touches += 1
                    reward -= 25
                    self.invincibility_timer = self.INVINCIBILITY_FRAMES
                    self._create_hit_particles(self.player_pos)
                    # SFX: Player hit sound
                    break # Only one hit per frame
        return reward

    def _create_hit_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": np.copy(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(10, 25),
                "color": self.COLOR_PLAYER
            })

    def _check_termination(self):
        return (
            self.souls_collected >= self.WIN_SOULS or
            self.ghoul_touches >= self.MAX_TOUCHES
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw gravestones
        for stone in self.gravestones:
            pygame.draw.rect(self.screen, self.COLOR_GRAVESTONE, stone)
            pygame.draw.rect(self.screen, tuple(int(c*0.8) for c in self.COLOR_GRAVESTONE), stone, 2)

        # Draw souls with glow
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
        glow_radius = self.SOUL_RADIUS + 3 * pulse
        for soul_pos in self.souls:
            pos_int = soul_pos.astype(int)
            # Simple glow by drawing a larger circle
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(glow_radius), self.COLOR_SOUL_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.SOUL_RADIUS, self.COLOR_SOUL)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.SOUL_RADIUS, self.COLOR_SOUL)

        # Draw ghouls with flicker
        flicker = self.steps % 10 > 5
        ghoul_color = self.COLOR_GHOUL if flicker else self.COLOR_GHOUL_FLICKER
        for ghoul in self.ghouls:
            pos_int = ghoul["pos"].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.GHOUL_RADIUS, ghoul_color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.GHOUL_RADIUS, ghoul_color)

        # Draw player
        is_invincible_blink = self.invincibility_timer > 0 and (self.steps // 3) % 2 == 0
        if not is_invincible_blink:
            player_color = self.COLOR_PLAYER_INVINCIBLE if self.invincibility_timer > 0 else self.COLOR_PLAYER
            pos_int = self.player_pos.astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, player_color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, player_color)

        # Draw particles
        for p in self.particles:
            pos_int = p['pos'].astype(int)
            size = max(1, int(p['lifespan'] / 5))
            pygame.draw.circle(self.screen, p['color'], pos_int, size)

    def _render_ui(self):
        # Souls collected
        soul_text = self.font_ui.render(f"Souls: {self.souls_collected}/{self.WIN_SOULS}", True, self.COLOR_TEXT_GOOD)
        self.screen.blit(soul_text, (10, 10))

        # Ghoul touches
        touch_text = self.font_ui.render(f"Hits: {self.ghoul_touches}/{self.MAX_TOUCHES}", True, self.COLOR_TEXT_BAD)
        text_rect = touch_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(touch_text, text_rect)

        # Game over message
        if self.game_over:
            if self.souls_collected >= self.WIN_SOULS:
                msg_text = self.font_msg.render("YOU WIN!", True, self.COLOR_TEXT_GOOD)
            else:
                msg_text = self.font_msg.render("GAME OVER", True, self.COLOR_TEXT_BAD)
            
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "souls_collected": self.souls_collected,
            "ghoul_touches": self.ghoul_touches,
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # The environment can be run headless
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # --- To play the game with a display ---
    # Create a new GameEnv with 'rgb_array' render mode
    # and use a separate pygame window for display.
    env_to_render = GameEnv()
    human_screen = pygame.display.set_mode((env_to_render.SCREEN_WIDTH, env_to_render.SCREEN_HEIGHT))
    pygame.display.set_caption("Haunted Graveyard")
    
    obs, info = env_to_render.reset()
    
    running = True
    while running:
        # Map pygame keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env_to_render.step(action)
        
        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env_to_render.reset()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env_to_render.reset()

        env_to_render.clock.tick(30) # Run at 30 FPS

    env_to_render.close()