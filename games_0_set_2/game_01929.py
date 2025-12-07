import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Space for a standard attack. Shift for a risky attack. Only one action (move or attack) per turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based tactical RPG. Position yourself on the grid and choose your attacks wisely to defeat the enemy before it defeats you."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 8
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.SCREEN_HEIGHT) // 2
        self.GRID_OFFSET_Y = 0
        self.TILE_SIZE = self.SCREEN_HEIGHT // self.GRID_SIZE
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_m = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_l = pygame.font.SysFont("Arial", 32, bold=True)

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_PLAYER = (60, 180, 255)
        self.COLOR_ENEMY = (255, 80, 80)
        self.COLOR_HEALTH_BG = (70, 40, 40)
        self.COLOR_HEALTH = (100, 220, 100)
        self.COLOR_ATTACK = (255, 255, 100)
        self.COLOR_TURN_INDICATOR = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)

        # Game constants
        self.MAX_HP = 50
        self.MAX_STEPS = 200 # Reduced from 1000 for faster episodes
        self.ENEMY_ATTACK_DAMAGE = 5
        self.STANDARD_ATTACK_RANGE = (5, 10)
        self.RISKY_ATTACK_RANGE = (0, 20)
        
        # State variables are initialized in reset()
        self.np_random = None
        self.player_pos = None
        self.enemy_pos = None
        self.player_hp = None
        self.enemy_hp = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.animations = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.animations = []
        
        # Place entities
        self.player_pos = self.np_random.integers(0, self.GRID_SIZE, size=2).tolist()
        while True:
            self.enemy_pos = self.np_random.integers(0, self.GRID_SIZE, size=2).tolist()
            if self.enemy_pos != self.player_pos:
                break
        
        self.player_hp = self.MAX_HP
        self.enemy_hp = self.MAX_HP
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        action_taken = False

        # --- PLAYER TURN ---
        # Priority: Attack > Move > Wait
        if shift_held or space_held:
            action_taken = True
            attack_type = "risky" if shift_held else "standard"
            if attack_type == "risky":
                damage = self.np_random.integers(self.RISKY_ATTACK_RANGE[0], self.RISKY_ATTACK_RANGE[1] + 1)
            else: # standard
                damage = self.np_random.integers(self.STANDARD_ATTACK_RANGE[0], self.STANDARD_ATTACK_RANGE[1] + 1)
            
            self.enemy_hp = max(0, self.enemy_hp - damage)
            reward += damage * 0.1
            self._add_animation("attack", self.enemy_pos, self.COLOR_ATTACK, duration=15)
        
        elif movement != 0:
            action_taken = True
            new_pos = list(self.player_pos)
            if movement == 1: new_pos[1] -= 1 # Up
            elif movement == 2: new_pos[1] += 1 # Down
            elif movement == 3: new_pos[0] -= 1 # Left
            elif movement == 4: new_pos[0] += 1 # Right

            if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
                self.player_pos = new_pos
            else:
                pass
        
        if not action_taken: # No-op or wait
            pass

        # --- ENEMY TURN (immediately follows player's turn) ---
        if self.enemy_hp > 0:
            damage_to_player = self.ENEMY_ATTACK_DAMAGE
            self.player_hp = max(0, self.player_hp - damage_to_player)
            reward -= damage_to_player * 0.1
            self._add_animation("attack", self.player_pos, self.COLOR_ENEMY, duration=15)

        # --- RESOLUTION ---
        self.steps += 1
        
        if self.player_hp <= 0:
            terminated = True
            reward = -100.0
            self.game_over = True
            text_data = {"text": "DEFEAT", "font": self.font_l, "color": self.COLOR_ENEMY}
            self._add_animation("text", (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2), text_data, duration=180)

        elif self.enemy_hp <= 0:
            terminated = True
            reward = 100.0
            self.game_over = True
            text_data = {"text": "VICTORY!", "font": self.font_l, "color": self.COLOR_PLAYER}
            self._add_animation("text", (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2), text_data, duration=180)

        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            text_data = {"text": "TIME UP", "font": self.font_l, "color": self.COLOR_TEXT}
            self._add_animation("text", (self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2), text_data, duration=180)

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        self._update_and_draw_animations()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_OFFSET_X + i * self.TILE_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.TILE_SIZE, self.GRID_OFFSET_Y + self.GRID_SIZE * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.TILE_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_SIZE * self.TILE_SIZE, self.GRID_OFFSET_Y + i * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw turn indicator
        if not self.game_over:
            px, py = self._grid_to_pixel(self.player_pos)
            indicator_rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
            self._draw_glowing_border(indicator_rect, self.COLOR_TURN_INDICATOR, 3, 20)

        # Draw entities
        self._draw_entity(self.player_pos, self.player_hp, self.COLOR_PLAYER, "Player")
        if self.enemy_hp > 0:
            self._draw_entity(self.enemy_pos, self.enemy_hp, self.COLOR_ENEMY, "Enemy")

    def _draw_entity(self, pos, hp, color, label):
        px, py = self._grid_to_pixel(pos)
        entity_rect = pygame.Rect(px + 5, py + 5, self.TILE_SIZE - 10, self.TILE_SIZE - 10)
        pygame.draw.rect(self.screen, color, entity_rect, border_radius=5)

        # Health bar
        bar_width = self.TILE_SIZE - 10
        bar_height = 8
        hp_ratio = max(0, hp / self.MAX_HP)
        
        bg_rect = pygame.Rect(px + 5, py - bar_height - 2, bar_width, bar_height)
        hp_rect = pygame.Rect(px + 5, py - bar_height - 2, int(bar_width * hp_ratio), bar_height)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect, border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, hp_rect, border_radius=2)
        
        # Health text
        hp_text = f"{int(hp)}/{self.MAX_HP}"
        self._draw_text(hp_text, self.font_s, self.COLOR_TEXT, (px + self.TILE_SIZE / 2, py - bar_height / 2 - 2))

    def _render_ui(self):
        # Left Panel
        self._draw_text("PLAYER", self.font_m, self.COLOR_PLAYER, (self.GRID_OFFSET_X / 2, 50))
        self._draw_text(f"HP: {int(self.player_hp)}/{self.MAX_HP}", self.font_s, self.COLOR_TEXT, (self.GRID_OFFSET_X / 2, 80))
        
        # Right Panel
        self._draw_text("ENEMY", self.font_m, self.COLOR_ENEMY, (self.SCREEN_WIDTH - self.GRID_OFFSET_X / 2, 50))
        self._draw_text(f"HP: {int(self.enemy_hp)}/{self.MAX_HP}", self.font_s, self.COLOR_TEXT, (self.SCREEN_WIDTH - self.GRID_OFFSET_X / 2, 80))

        # Bottom Panel
        self._draw_text(f"Turn: {self.steps}", self.font_m, self.COLOR_TEXT, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20))
        
        if not self.game_over:
            self._draw_text("YOUR TURN", self.font_m, self.COLOR_TEXT, (self.SCREEN_WIDTH / 2, 30))
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_hp": self.player_hp,
            "enemy_hp": self.enemy_hp
        }

    # --- Helper Functions ---
    
    def _grid_to_pixel(self, grid_pos):
        px = self.GRID_OFFSET_X + grid_pos[0] * self.TILE_SIZE
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.TILE_SIZE
        return px, py

    def _draw_text(self, text, font, color, pos, centered=True):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, color)
        
        shadow_rect = shadow_surf.get_rect()
        text_rect = text_surf.get_rect()

        if centered:
            shadow_rect.center = (pos[0] + 1, pos[1] + 1)
            text_rect.center = pos
        else:
            shadow_rect.topleft = (pos[0] + 1, pos[1] + 1)
            text_rect.topleft = pos
            
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)
        
    def _add_animation(self, anim_type, pos, data, duration=30):
        self.animations.append({
            "type": anim_type,
            "pos": pos,
            "data": data,
            "duration": duration,
            "timer": duration
        })

    def _update_and_draw_animations(self):
        active_animations = []
        for anim in self.animations:
            anim["timer"] -= 1
            if anim["timer"] > 0:
                self._draw_animation(anim)
                active_animations.append(anim)
        self.animations = active_animations

    def _draw_animation(self, anim):
        progress = 1.0 - (anim["timer"] / anim["duration"])
        if anim["type"] == "attack":
            px, py = self._grid_to_pixel(anim["pos"])
            center_x = px + self.TILE_SIZE // 2
            center_y = py + self.TILE_SIZE // 2
            color = anim["data"]
            
            radius = int(progress * self.TILE_SIZE * 0.7)
            alpha = int(255 * (1 - progress))
            
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, (*color, alpha))
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, (*color, alpha))
        
        elif anim["type"] == "text":
            text_info = anim["data"]
            self._draw_text(
                text=text_info["text"],
                font=text_info["font"],
                color=text_info["color"],
                pos=anim["pos"],
                centered=True
            )

    def _draw_glowing_border(self, rect, color, width, alpha):
        for i in range(width):
            alpha_val = int(alpha * (1 - i / width))
            if alpha_val < 0: alpha_val = 0
            temp_color = (*color, alpha_val)
            temp_rect = rect.inflate(i * 2, i * 2)
            try:
                pygame.draw.rect(self.screen, temp_color, temp_rect, 1, border_radius=6)
            except TypeError: # Older pygame versions might not support alpha in this call
                # Fallback without alpha, less pretty but won't crash
                pygame.draw.rect(self.screen, color, temp_rect, 1, border_radius=6)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # You might need to `pip install pygame`
    # Un-comment the next line to run in a window instead of headless
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print("--- Tactical Grid RPG ---")
    print(env.game_description)
    print(env.user_guide)
    
    # Use a dummy screen for display if not already headless
    try:
        display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Tactical Grid RPG")
        is_display_enabled = True
    except pygame.error:
        print("\nPygame display unavailable (running headless). Gameplay will proceed without a window.")
        is_display_enabled = False

    action = [0, 0, 0] # No-op
    
    running = True
    while running:
        if is_display_enabled:
            # Human input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Since it's turn-based, we register one action and then step
                if event.type == pygame.KEYDOWN:
                    current_action = [0, 0, 0] # Reset action
                    if event.key == pygame.K_UP:
                        current_action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        current_action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        current_action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        current_action[0] = 4
                    elif event.key == pygame.K_SPACE:
                        current_action[1] = 1
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                        current_action[2] = 1
                    elif event.key == pygame.K_r: # Reset key
                        print("\n--- Resetting Game ---")
                        obs, info = env.reset()
                        done = False
                        continue
                    
                    # If any valid key was pressed, step the environment
                    if any(current_action):
                        if done:
                            print("Game is over. Press 'R' to reset.")
                        else:
                            obs, reward, terminated, truncated, info = env.step(current_action)
                            done = terminated or truncated
                            print(f"Step: {info['steps']}, Score: {info['score']:.1f}, Reward: {reward:.1f}, Player HP: {info['player_hp']}, Enemy HP: {info['enemy_hp']}")

            # Render the observation to the display screen
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()

            if done:
                if 'game_over_printed' not in locals() or not game_over_printed:
                    print("Game Over! Press 'R' to play again or close the window.")
                    game_over_printed = True
            else:
                game_over_printed = False
        else:
            # If no display, just end the script
            print("Running in headless mode. Main loop is for interactive play. Exiting.")
            running = False

    env.close()